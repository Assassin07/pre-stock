"""
Google Colab专用配置文件
针对免费GPU环境优化的参数设置
"""

# 数据配置 - Colab优化版
DATA_CONFIG = {
    'sequence_length': 30,  # 减少序列长度以节省内存（原60）
    'prediction_days': 3,   # 减少预测天数以加快训练（原5）
    'train_ratio': 0.8,     # 训练集比例
    'val_ratio': 0.1,       # 验证集比例
    'test_ratio': 0.1,      # 测试集比例
}

# 模型配置 - Colab优化版
MODEL_CONFIG = {
    'input_size': 20,       # 输入特征数量
    'hidden_size': 64,      # 减少隐藏层大小以节省内存（原128）
    'num_layers': 2,        # 减少层数以加快训练（原3）
    'dropout': 0.2,         # Dropout率
    'bidirectional': True,  # 是否使用双向LSTM
}

# 训练配置 - Colab优化版
TRAINING_CONFIG = {
    'batch_size': 16,       # 减少批次大小以节省内存（原32）
    'learning_rate': 0.002, # 稍微提高学习率以加快收敛（原0.001）
    'num_epochs': 30,       # 减少训练轮数以节省时间（原100）
    'patience': 5,          # 减少早停耐心值（原10）
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

# Colab专用设置
COLAB_CONFIG = {
    'use_mixed_precision': True,    # 使用混合精度训练以节省内存
    'gradient_clip_val': 0.5,       # 梯度裁剪值
    'save_every_n_epochs': 5,       # 每N个epoch保存一次模型
    'max_data_points': 1000,        # 限制最大数据点数以节省内存
    'enable_progress_bar': True,    # 启用进度条
    'auto_download_results': True,  # 自动下载结果
}

# 快速模式配置（用于演示和测试）
QUICK_MODE_CONFIG = {
    'sequence_length': 20,
    'prediction_days': 3,
    'hidden_size': 32,
    'num_layers': 1,
    'batch_size': 8,
    'num_epochs': 10,
    'patience': 3,
}

# 性能模式配置（如果有足够的GPU资源）
PERFORMANCE_MODE_CONFIG = {
    'sequence_length': 60,
    'prediction_days': 7,
    'hidden_size': 128,
    'num_layers': 3,
    'batch_size': 32,
    'num_epochs': 100,
    'patience': 10,
}
