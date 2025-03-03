# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
#import streamlit as st

def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
# 自定义损失函数
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true.long(), num_classes=2).float()
        return self.loss_fn(y_pred, y_true)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, X):
        X = X + self.pe[:, :X.size(1)]
        return self.dropout(X)

# Transformer 模型定义
class TransformerClassifier(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes=2, 
                 hidden_dim=2048,
                 nhead=20, 
                 num_encoder_layers=10,
                 dropout=0.1,
                 window_size=30):
        """
        改进后的 Transformer 模型，在编码器后增加额外的多头自注意力层，并加入残差连接和层归一化。
        """
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(num_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dropout=dropout,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.attn_layernorm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.float()
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_layernorm(x + attn_out)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# MLP 模型定义
class MLPClassifierModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5):
        super(MLPClassifierModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        X = self.fc1(X)
        X = self.activation(X)
        X = self.dropout(X)
        X = self.fc2(X)
        return X

# 计算类别权重
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)

# 构造 Transformer 分类器
def get_transformer_classifier(num_features, window_size, class_weights=None):
    if class_weights is not None:
        loss = WeightedCrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    net = NeuralNetClassifier(
        module=TransformerClassifier,
        module__num_features=num_features,
        module__window_size=window_size,
        module__hidden_dim=512,
        module__nhead=8,
        module__num_encoder_layers=3,
        module__dropout=0.1,
        max_epochs=100,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        criterion=loss,
        batch_size=128,
        train_split=None,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return ('transformer', net)

# 构造 MLP 分类器
def get_mlp_classifier(input_dim, class_weights=None):
    if class_weights is not None:
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.float()
        loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    net = NeuralNetClassifier(
        module=MLPClassifierModule,
        module__input_dim=input_dim,
        module__hidden_dim=64,
        module__output_dim=2,
        module__dropout=0.5,
        criterion=loss,
        optimizer=torch.optim.Adam,
        max_epochs=100,
        lr=1e-3,
        batch_size=64,
        train_split=None,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return ('mlp', net)

# 仅支持 Transformer 与 MLP，其他模型不再提供
def get_classifier(classifier_name, num_features=None, window_size=10, class_weight=None):
    if classifier_name == 'Transformer':
        if num_features is None:
            raise ValueError("num_features必须为Transformer模型指定")
        return get_transformer_classifier(num_features, window_size, class_weights=class_weight)
    elif classifier_name == 'MLP':
        if num_features is None:
            raise ValueError("num_features必须为MLP模型指定")
        return get_mlp_classifier(num_features, class_weights=class_weight)
    else:
        raise ValueError(f"未知的分类器名称: {classifier_name}. 目前仅支持 Transformer 和 MLP。")
