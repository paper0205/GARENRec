import os
import torch
import numpy as np
import random
import yaml
import argparse
from pathlib import Path

def set_seed(seed):
    """
    모든 난수 생성기의 시드를 설정하여 재현성 확보
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_config(config_path):
    """
    YAML 파일에서 설정 로드
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device():
    """
    사용 가능한 디바이스 반환
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_graph_data(champion_to_idx, relationships):
    """
    그래프 데이터 준비
    """
    edge_index = []
    edge_weight = []
    for champ_a, champ_b, weight in relationships:
        if champ_a in champion_to_idx and champ_b in champion_to_idx:
            edge_index.append([champion_to_idx[champ_a], champion_to_idx[champ_b]])
            edge_weight.append(weight)
            # 양방향 그래프로 만들기
            edge_index.append([champion_to_idx[champ_b], champion_to_idx[champ_a]])
            edge_weight.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_index, edge_weight

def create_champion_mapping(data):
    """
    챔피언 이름을 인덱스로 매핑
    NaN값은 무시하고 실제 챔피언만 매핑
    """
    champions = set()
    draft_columns = ['BB1','RB1','BB2','RB2','BB3','RB3',
                    'BP1','RP1','RP2','BP2','BP3','RP3',
                    'RB4','BB4','RB5','BB5',
                    'RP4','BP4','BP5','RP5']

    for col in draft_columns:
        # NaN이 아닌 값만 추가
        valid_champs = data[col].dropna().unique()
        champions.update(valid_champs)

    # 특별한 토큰 추가
    champions.add('NO_BAN')  # 밴을 하지 않은 경우를 위한 특별 토큰

    champion_to_idx = {champ: idx for idx, champ in enumerate(sorted(champions))}
    idx_to_champion = {idx: champ for champ, idx in champion_to_idx.items()}

    return champion_to_idx, idx_to_champion

def create_position_mapping():
    """
    포지션을 인덱스로 매핑
    """
    positions = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    position_to_idx = {pos: idx for idx, pos in enumerate(positions)}
    idx_to_position = {idx: pos for pos, idx in position_to_idx.items()}

    return position_to_idx, idx_to_position

def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps=0, last_epoch=-1):
    """
    코사인 스케줄러 구현
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + cos(pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(optimizer, config, num_training_steps):
    """
    학습률 스케줄러 생성
    """
    scheduler_type = config['training']['scheduler']
    
    if scheduler_type == 'cosine':
        # 코사인 스케줄러
        num_warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )
    elif scheduler_type == 'step':
        # 스텝 스케줄러
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=5, 
            gamma=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def save_model(model, path, epoch, optimizer=None, scheduler=None, best_score=None):
    """
    모델과 훈련 상태 저장
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    if best_score is not None:
        checkpoint['best_score'] = best_score
        
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None, scheduler=None):
    """
    저장된 모델과 훈련 상태 로드
    """
    if not os.path.exists(path):
        print(f"No model found at {path}")
        return model, 0, None
    
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_score = checkpoint.get('best_score', None)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded from {path} (epoch: {epoch})")
    return model, epoch, best_score

def parse_args():
    """
    명령줄 인수 파싱
    """
    parser = argparse.ArgumentParser(description="GARENRec: LoL Champion Recommendation System")
    parser.add_argument('--config', type=str, default='./config/config.yaml', 
                        help='Path to the config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'infer'], default='train',
                        help='Run mode: train, test, or inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint to load (for test/inference)')
    return parser.parse_args()

from math import cos, pi  # get_cosine_schedule_with_warmup 함수에서 사용