import torch
import numpy as np
import torch.nn.functional as F

def calculate_hr(predictions, targets, k):
    """
    Hit Ratio@k 계산
    predictions: (batch_size, num_classes) 형태의 로짓
    targets: (batch_size,) 형태의 실제 라벨
    k: 상위 k개 항목 고려
    """
    batch_size = predictions.size(0)
    probabilities = F.softmax(predictions, dim=1)
    _, top_indices = torch.topk(probabilities, k=k, dim=1)

    # targets를 (batch_size, 1)로 확장하여 broadcasting
    targets_expanded = targets.unsqueeze(-1)

    # top k 내에 정답이 있는지 확인 (batch_size, k)
    hits = (top_indices == targets_expanded).any(dim=1).float()

    # 배치에 대한 평균 계산
    return hits.mean().item()

def calculate_ndcg(predictions, targets, k):
    """
    NDCG@k 계산
    predictions: (batch_size, num_classes) 형태의 로짓
    targets: (batch_size,) 형태의 실제 라벨
    k: 상위 k개 항목 고려
    """
    batch_size = predictions.size(0)

    # 예측값을 확률로 변환
    probabilities = F.softmax(predictions, dim=1)

    # 상위 k개 인덱스와 확률값 가져오기
    top_probs, top_indices = torch.topk(probabilities, k=min(k, probabilities.size(1)), dim=1)

    # DCG와 IDCG 계산
    dcg = torch.zeros(batch_size, device=predictions.device)
    idcg = torch.zeros(batch_size, device=predictions.device)

    for i in range(k):
        # DCG 계산
        rel = (top_indices[:, i] == targets).float()
        pos = i + 1
        dcg += rel / torch.log2(torch.tensor(pos + 1, dtype=torch.float))

        # IDCG 계산 (이상적인 경우 첫 번째 위치에 정답이 있음)
        if i == 0:
            idcg += 1.0  # rel=1 for ideal case

    # NDCG 계산
    ndcg = dcg / idcg
    # 0으로 나누는 경우(idcg=0) 처리
    ndcg = torch.where(idcg > 0, ndcg, torch.zeros_like(ndcg))

    return ndcg.mean().item()

def evaluate_model(model, data_loader, edge_index, edge_weight, device):
    """
    모델 평가 함수
    반환: loss, hr_scores, ndcg_scores
    """
    model.eval()
    total_loss = 0
    hr_scores = {k: 0.0 for k in range(1, 6)}  # HR@1 ~ HR@5
    ndcg_scores = {k: 0.0 for k in range(1, 6)}  # NDCG@1 ~ NDCG@5
    criterion = MaskedCrossEntropyLoss()
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
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
            total_loss += loss.item()

            # 마지막 예측에 대한 평가
            last_predictions = champion_logits[:, -1, :]
            last_targets = champions[:, -1]
            last_available = available_mask[:, -1, :]

            # 마스킹된 logit으로 평가 지표 계산
            masked_predictions = last_predictions + (1 - last_available) * -1e4

            # HR과 NDCG 계산
            for k in range(1, 6):
                hr_scores[k] += calculate_hr(masked_predictions, last_targets, k)
                ndcg_scores[k] += calculate_ndcg(masked_predictions, last_targets, k)

            num_batches += 1

    # 평균 계산
    avg_loss = total_loss / num_batches
    for k in range(1, 6):
        hr_scores[k] /= num_batches
        ndcg_scores[k] /= num_batches

    return avg_loss, hr_scores, ndcg_scores

class MaskedCrossEntropyLoss(torch.nn.Module):
    """
    마스킹된 크로스 엔트로피 손실 함수
    이미 선택된 챔피언은 다시 선택할 수 없도록 마스킹
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, available_mask):
        masked_logits = logits + (1 - available_mask) * -1e6  # NaN 방지용 큰 음수 사용
        log_probs = F.log_softmax(masked_logits, dim=-1)

        if len(logits.shape) == 3:
            targets = targets.unsqueeze(-1)
            loss = -torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
            loss = loss * (available_mask.sum(dim=-1) > 0).float()
            return loss.mean()
        else:
            loss = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            return loss.mean()