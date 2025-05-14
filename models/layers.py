import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GraphEmbeddingLayer(nn.Module):
    """
    그래프 임베딩 레이어
    챔피언 간의 관계를 그래프로 표현하여 임베딩 생성
    """
    def __init__(self, num_champions, embedding_dim, gnn_hidden_dim, heads=4, concat=True):
        super(GraphEmbeddingLayer, self).__init__()
        self.champion_embeddings = nn.Embedding(num_champions, embedding_dim)

        # GATConv를 edge_weight를 사용하는 버전으로 수정
        self.gnn = pyg_nn.GATConv(
            in_channels=embedding_dim,
            out_channels=gnn_hidden_dim // heads,
            heads=heads,
            concat=concat,
            edge_dim=1  # edge_weight를 사용하기 위해 추가
        )

        # GNN 출력 크기를 다시 임베딩 크기로 맞춤
        if concat:
            self.output_projection = nn.Linear(gnn_hidden_dim, embedding_dim)
        else:
            self.output_projection = nn.Linear(gnn_hidden_dim // heads, embedding_dim)

    def forward(self, edge_index, edge_weight=None):
        node_features = self.champion_embeddings.weight
        # edge_weight를 [E, 1] 형태로 변환
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        graph_features = self.gnn(node_features, edge_index, edge_attr=edge_attr)
        output_features = self.output_projection(graph_features)
        return output_features

class MaskedCrossEntropyLoss(nn.Module):
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
