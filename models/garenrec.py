import torch
import torch.nn as nn

from models.layers import GraphEmbeddingLayer

class GARENRec(nn.Module):
    """
    GARENRec: Graph Attentional Recommendation Engine for Networks
    
    LOL 챔피언 추천을 위한 그래프 기반 트랜스포머 모델
    """
    def __init__(self, num_champions, num_positions=5, embedding_dim=128,
                 gnn_hidden_dim=64, nhead=4, dropout=0.2, feature_dim=18, num_layers=2):
        super(GARENRec, self).__init__()

        # 임베딩 레이어들
        self.graph_embedding_layer = GraphEmbeddingLayer(
            num_champions, 
            embedding_dim, 
            gnn_hidden_dim
        )
        self.position_embeddings = nn.Embedding(num_positions, embedding_dim)
        self.step_embeddings = nn.Embedding(20, embedding_dim)
        self.team_embeddings = nn.Embedding(2, embedding_dim)

        # 매치 피처를 위한 새로운 레이어
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력 레이어
        self.champion_predictor = nn.Linear(embedding_dim, num_champions)

        # 드래프트 순서 정의
        self.draft_order = ['BB1','RB1','BB2','RB2','BB3','RB3',
                           'BP1','RP1','RP2','BP2','BP3','RP3',
                           'RB4','BB4','RB5','BB5',
                           'RP4','BP4','BP5','RP5']

    def forward(self, champion_sequence, position_sequence=None, mask_sequence=None,
               edge_index=None, edge_weight=None, feature_sequence=None):
        """
        Forward 함수
        
        Args:
            champion_sequence: [batch_size, seq_len] 형태의 챔피언 인덱스 시퀀스
            position_sequence: [batch_size, num_picks] 형태의 포지션 인덱스 시퀀스
            mask_sequence: [batch_size, seq_len] 형태의 픽/밴 마스크 시퀀스
            edge_index: 그래프의 에지 인덱스
            edge_weight: 그래프의 에지 가중치
            feature_sequence: [batch_size, seq_len, feature_dim] 형태의 매치 피처 시퀀스
            
        Returns:
            [batch_size, seq_len, num_champions] 형태의 챔피언 예측 로짓
        """
        batch_size, seq_len = champion_sequence.size()

        # 그래프 임베딩
        graph_embeddings = self.graph_embedding_layer(edge_index, edge_weight)
        x = graph_embeddings[champion_sequence]

        # 스텝 임베딩
        steps = torch.arange(seq_len, device=champion_sequence.device)
        x = x + self.step_embeddings(steps.unsqueeze(0).expand(batch_size, -1))

        # 팀 정보 추가
        team_indices = torch.zeros(seq_len, device=champion_sequence.device, dtype=torch.long)
        for i in range(seq_len):
            if self.draft_order[i].startswith('R'):
                team_indices[i] = 1

        team_embed = self.team_embeddings(team_indices.unsqueeze(0).expand(batch_size, -1))
        x = x + team_embed

        # 포지션 임베딩
        if position_sequence is not None and mask_sequence is not None:
            position_embed = torch.zeros_like(x)
            pick_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            for i in range(seq_len):
                is_pick = mask_sequence[:, i]
                if is_pick.any():
                    batch_mask = is_pick
                    current_pick_idx = pick_idx[batch_mask]
                    current_pick_idx = current_pick_idx.clamp(0, position_sequence.size(1) - 1)
                    pos_embed = self.position_embeddings(
                        position_sequence[batch_mask, current_pick_idx]
                    )
                    position_embed[batch_mask, i] = pos_embed
                    pick_idx[batch_mask] += 1
            x = x + position_embed

        # 매치 피처 추가
        if feature_sequence is not None:
            feature_embeddings = self.feature_projection(feature_sequence)
            x = x + feature_embeddings

        # Transformer 처리
        x = self.transformer(x)
        champion_logits = self.champion_predictor(x)

        return champion_logits
