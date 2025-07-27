"""
配置文件
"""

# 数据配置
DATA_CONFIG = {
    'sequence_length': 60,  # 输入序列长度（天数）
    'prediction_days': 5,   # 预测天数
    'train_ratio': 0.8,     # 训练集比例
    'val_ratio': 0.1,       # 验证集比例
    'test_ratio': 0.1,      # 测试集比例
}

# 模型配置
MODEL_CONFIG = {
    'input_size': 20,       # 输入特征数量
    'hidden_size': 128,     # LSTM隐藏层大小
    'num_layers': 3,        # LSTM层数
    'dropout': 0.2,         # Dropout率
    'bidirectional': True,  # 是否使用双向LSTM
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 10,         # 早停耐心值
    'weight_decay': 1e-5,   # L2正则化
}

# 数据路径
PATHS = {
    'data_dir': 'data/',
    'model_dir': 'models/',
    'results_dir': 'results/',
}

# 股票代码示例
DEFAULT_STOCK_CODE = '000001'  # 平安银行
