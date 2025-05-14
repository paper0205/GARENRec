import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import logging
from pathlib import Path

from models.garenrec import GARENRec
from data.dataset import DraftDatasetWithFeatures, prepare_data
from data.match_data_processor import MatchDataProcessor
from data.champion_relationships import ChampionRelationships
from utils.metrics import evaluate_model, MaskedCrossEntropyLoss
from utils.utils import (
    set_seed, 
    load_config, 
    get_device, 
    prepare_graph_data,
    create_champion_mapping,
    create_position_mapping,
    get_scheduler,
    save_model,
    load_model,
    parse_args
)

def setup_logging(config):
    """로깅 설정"""
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train(config, logger):
    """
    모델 학습 함수
    """
    # 디바이스 설정
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 시드 설정
    seed = config['data']['random_seed']
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    # 데이터 로드
    logger.info("Loading data...")
    draft_data_path = config['data']['draft_data_path']
    match_data_path = config['data']['match_data_path']
    
    all_draft_data = pd.read_csv(draft_data_path)
    draft_data, val_data, test_data = prepare_data(
        draft_data_path,
        test_size=config['data']['test_ratio'],
        val_size=config['data']['val_ratio'],
        random_state=seed
    )
    match_data = pd.read_csv(match_data_path)
    
    # 챔피언 및 포지션 매핑 생성
    logger.info("Creating champion and position mappings...")
    champion_to_idx, idx_to_champion = create_champion_mapping(all_draft_data)
    position_to_idx, idx_to_position = create_position_mapping()
    
    # 챔피언 관계 분석
    logger.info("Analyzing champion relationships...")
    champion_relationships = ChampionRelationships(draft_data, champion_to_idx)
    
    # 매치 데이터 프로세서 초기화 및 전처리
    logger.info("Preprocessing match data...")
    match_processor = MatchDataProcessor()
    processed_match_data = match_processor.preprocess_once(match_data)
    
    # 데이터셋 생성
    logger.info("Creating datasets...")
    train_dataset = DraftDatasetWithFeatures(
        draft_data, champion_to_idx, position_to_idx, processed_match_data)
    val_dataset = DraftDatasetWithFeatures(
        val_data, champion_to_idx, position_to_idx, processed_match_data)
    test_dataset = DraftDatasetWithFeatures(
        test_data, champion_to_idx, position_to_idx, processed_match_data)
    
    # 데이터 로더 생성
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 생성
    logger.info("Initializing model...")
    model = GARENRec(
        num_champions=len(champion_to_idx),
        num_positions=len(position_to_idx),
        embedding_dim=config['model']['embedding_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        nhead=config['model']['nhead'],
        dropout=config['model']['dropout'],
        feature_dim=config['model']['feature_dim'],
        num_layers=config['model']['num_layers']
    )
    model = model.to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = MaskedCrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 스케줄러 설정
    num_training_steps = len(train_loader) * config['training']['max_epochs']
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # 그래프 데이터 준비
    edge_index, edge_weight = prepare_graph_data(
        champion_relationships.champion_to_idx,
        champion_relationships.relationships
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # 체크포인트 디렉토리 생성
    save_dir = Path(config['paths']['model_save_path'])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / 'best_model.pt'
    last_model_path = save_dir / 'last_model.pt'
    
    # 학습 시작
    logger.info("Starting training...")
    max_epochs = config['training']['max_epochs']
    patience = config['training']['patience']
    gradient_clip = config['training']['gradient_clip']
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            champions = batch['draft_sequence'].to(device)
            positions = batch['position_sequence'].to(device)
            mask = batch['mask_sequence'].to(device)
            features = batch['feature_sequence'].to(device)
            
            champion_logits = model(
                champions[:, :-1],
                positions,
                mask[:, :-1],
                edge_index,
                edge_weight,
                features[:, :-1]
            )
            
            batch_size, seq_len = champion_logits.size()[:2]
            num_champions = champion_logits.size(-1)
            available_mask = torch.ones((batch_size, seq_len, num_champions), device=device)
            
            for b in range(batch_size):
                for t in range(seq_len):
                    selected_champions = champions[b, :t+1]
                    available_mask[b, t, selected_champions] = 0
            
            champion_loss = criterion(
                champion_logits,
                champions[:, 1:],
                available_mask
            )
            
            loss = champion_loss
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / num_batches
        
        # Validation
        val_loss, val_hr, val_ndcg = evaluate_model(model, val_loader, edge_index, edge_weight, device)
        
        # 로깅
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Validation HR@5: {val_hr[5]:.4f}, NDCG@5: {val_ndcg[5]:.4f}")
        
        # 모델 저장
        save_model(model, last_model_path, epoch, optimizer, scheduler)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, best_model_path, epoch, optimizer, scheduler, best_val_loss)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # 최종 평가
    logger.info("Loading best model for evaluation...")
    model, _, _ = load_model(model, best_model_path)
    
    logger.info("Evaluating on test set...")
    test_loss, test_hr, test_ndcg = evaluate_model(model, test_loader, edge_index, edge_weight, device)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info("Test Metrics:")
    for k in range(1, 6):
        logger.info(f"  HR@{k}: {test_hr[k]:.4f}, NDCG@{k}: {test_ndcg[k]:.4f}")
    
    return model, test_hr, test_ndcg

def test(config, logger, checkpoint_path=None):
    """
    학습된 모델 테스트 함수
    """
    # 디바이스 설정
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 시드 설정
    seed = config['data']['random_seed']
    set_seed(seed)
    
    # 데이터 로드
    logger.info("Loading data...")
    draft_data_path = config['data']['draft_data_path']
    match_data_path = config['data']['match_data_path']
    
    all_draft_data = pd.read_csv(draft_data_path)
    _, _, test_data = prepare_data(
        draft_data_path,
        test_size=config['data']['test_ratio'],
        val_size=config['data']['val_ratio'],
        random_state=seed
    )
    match_data = pd.read_csv(match_data_path)
    
    # 챔피언 및 포지션 매핑 생성
    logger.info("Creating champion and position mappings...")
    champion_to_idx, idx_to_champion = create_champion_mapping(all_draft_data)
    position_to_idx, idx_to_position = create_position_mapping()
    
    # 챔피언 관계 분석
    logger.info("Analyzing champion relationships...")
    champion_relationships = ChampionRelationships(all_draft_data, champion_to_idx)
    
    # 매치 데이터 프로세서 초기화 및 전처리
    logger.info("Preprocessing match data...")
    match_processor = MatchDataProcessor()
    processed_match_data = match_processor.preprocess_once(match_data)
    
    # 테스트 데이터셋 생성
    logger.info("Creating test dataset...")
    test_dataset = DraftDatasetWithFeatures(
        test_data, champion_to_idx, position_to_idx, processed_match_data)
    
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    # 모델 생성
    logger.info("Initializing model...")
    model = GARENRec(
        num_champions=len(champion_to_idx),
        num_positions=len(position_to_idx),
        embedding_dim=config['model']['embedding_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        nhead=config['model']['nhead'],
        dropout=config['model']['dropout'],
        feature_dim=config['model']['feature_dim'],
        num_layers=config['model']['num_layers']
    )
    
    # 모델 로드
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['paths']['model_save_path'], 'best_model.pt')
    
    logger.info(f"Loading model from {checkpoint_path}...")
    model, _, _ = load_model(model, checkpoint_path)
    model = model.to(device)
    
    # 그래프 데이터 준비
    edge_index, edge_weight = prepare_graph_data(
        champion_relationships.champion_to_idx,
        champion_relationships.relationships
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # 평가
    logger.info("Evaluating model...")
    test_loss, test_hr, test_ndcg = evaluate_model(model, test_loader, edge_index, edge_weight, device)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info("Test Metrics:")
    for k in range(1, 6):
        logger.info(f"  HR@{k}: {test_hr[k]:.4f}, NDCG@{k}: {test_ndcg[k]:.4f}")
    
    return test_hr, test_ndcg

def main():
    """
    메인 함수
    """
    # 인자 파싱
    args = parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    logger = setup_logging(config)
    logger.info(f"Starting GARENRec with config: {args.config}")
    
    # 모드에 따라 실행
    if args.mode == 'train':
        logger.info("Running in training mode")
        train(config, logger)
    elif args.mode == 'test':
        logger.info("Running in test mode")
        test(config, logger, args.checkpoint)
    else:  # 'infer' 모드
        logger.info("Running in inference mode")
        # 추론 모드 구현 (필요시)
        logger.info("Inference mode not implemented yet")

if __name__ == "__main__":
    main()
