"""
深度学习模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
        """
        LSTM股票预测模型
        
        Args:
            input_size: 输入特征数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出大小（预测天数）
            dropout: Dropout率
            bidirectional: 是否使用双向LSTM
        """
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 计算LSTM输出大小
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 预测结果 (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        last_output = attn_out[:, -1, :]
        
        # 全连接层
        output = self.fc_layers(last_output)
        
        return output


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
        """
        GRU股票预测模型
        
        Args:
            input_size: 输入特征数量
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            output_size: 输出大小（预测天数）
            dropout: Dropout率
            bidirectional: 是否使用双向GRU
        """
        super(StockGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 计算GRU输出大小
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, gru_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 4, output_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 预测结果 (batch_size, output_size)
        """
        # GRU前向传播
        gru_out, hidden = self.gru(x)
        
        # 使用最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # 全连接层
        output = self.fc_layers(last_output)
        
        return output


class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        """
        Transformer股票预测模型
        
        Args:
            input_size: 输入特征数量
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            output_size: 输出大小（预测天数）
            dropout: Dropout率
        """
        super(StockTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 预测结果 (batch_size, output_size)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        transformer_out = self.transformer(x)
        
        # 使用最后一个时间步的输出
        last_output = transformer_out[:, -1, :]
        
        # 输出投影
        output = self.output_projection(last_output)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """位置编码"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_model(model_type='lstm', input_size=None, output_size=None):
    """
    创建模型
    
    Args:
        model_type: 模型类型 ('lstm', 'gru', 'transformer')
        input_size: 输入特征数量
        output_size: 输出大小
        
    Returns:
        nn.Module: 创建的模型
    """
    if input_size is None:
        input_size = MODEL_CONFIG['input_size']
    if output_size is None:
        output_size = 5  # 默认预测5天
    
    if model_type.lower() == 'lstm':
        model = StockLSTM(
            input_size=input_size,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            output_size=output_size,
            dropout=MODEL_CONFIG['dropout'],
            bidirectional=MODEL_CONFIG['bidirectional']
        )
    elif model_type.lower() == 'gru':
        model = StockGRU(
            input_size=input_size,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            output_size=output_size,
            dropout=MODEL_CONFIG['dropout'],
            bidirectional=MODEL_CONFIG['bidirectional']
        )
    elif model_type.lower() == 'transformer':
        model = StockTransformer(
            input_size=input_size,
            d_model=MODEL_CONFIG['hidden_size'],
            nhead=8,
            num_layers=MODEL_CONFIG['num_layers'],
            output_size=output_size,
            dropout=MODEL_CONFIG['dropout']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model
