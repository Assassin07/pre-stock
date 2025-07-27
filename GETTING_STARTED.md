# 🚀 快速开始指南

## 📋 系统要求

- Python 3.7+
- 8GB+ 内存（推荐）
- 网络连接（可选，支持离线模式）

## 🔧 安装步骤

### 步骤1：克隆或下载项目
```bash
# 如果使用git
git clone <repository-url>
cd stock-prediction

# 或直接下载zip文件并解压
```

### 步骤2：安装依赖
```bash
# 推荐：使用智能安装脚本
python install_dependencies.py

# 或手动安装
pip install torch numpy pandas matplotlib plotly seaborn scikit-learn akshare tqdm joblib ta
```

### 步骤3：运行测试
```bash
# 快速测试（推荐）
python quick_test.py

# 完整测试
python test_system.py

# 网络问题诊断
python fix_network_issues.py
```

## 🎯 使用方法

### 方法1：快速预测（推荐新手）
```bash
# 预测平安银行未来5天走势
python main.py --stock_code 000001 --mode both --days 5

# 预测其他股票
python main.py --stock_code 600519 --mode both --days 3  # 贵州茅台
```

### 方法2：分步骤使用
```bash
# 1. 只训练模型
python main.py --stock_code 000001 --mode train

# 2. 只进行预测（需要先训练）
python main.py --stock_code 000001 --mode predict --days 5

# 3. 使用不同模型
python main.py --stock_code 000001 --model_type gru --days 3
```

### 方法3：使用示例脚本
```bash
python example.py
```

### 方法4：Google Colab
1. 上传 `Stock_Prediction_Colab.ipynb` 到Colab
2. 按照notebook中的步骤执行

## 📊 常用股票代码

| 代码 | 名称 | 代码 | 名称 |
|------|------|------|------|
| 000001 | 平安银行 | 600036 | 招商银行 |
| 000002 | 万科A | 600519 | 贵州茅台 |
| 000858 | 五粮液 | 002415 | 海康威视 |

## 🛠️ 故障排除

### 问题1：依赖安装失败
```bash
# 解决方案1：使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ <package-name>

# 解决方案2：更新pip
python -m pip install --upgrade pip

# 解决方案3：使用conda
conda install <package-name>
```

### 问题2：股票数据获取失败
```bash
# 诊断网络问题
python fix_network_issues.py

# 系统会自动使用示例数据，不影响功能测试
```

### 问题3：内存不足
```bash
# 使用Colab配置（减少内存使用）
python main_colab.py
```

### 问题4：GPU相关错误
```bash
# 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
python main.py --stock_code 000001 --mode both --days 3
```

## 📈 预测结果解读

### 输出示例
```
📊 预测结果:
当前价格: 12.34
第1天 (2024-01-15): 12.45 (+0.11, +0.89%) 📈
第2天 (2024-01-16): 12.38 (-0.07, -0.56%) 📉
第3天 (2024-01-17): 12.52 (+0.14, +1.13%) 📈

📊 总体趋势分析:
🟢 看涨 (+1.46%)
```

### 技术信号解读
- **RSI**: 超买(>70) / 超卖(<30) / 中性
- **MACD**: 看涨 / 看跌
- **MA**: 看涨(短期均线在长期均线上方) / 看跌

### 评估指标说明
- **RMSE**: 均方根误差，越小越好
- **MAE**: 平均绝对误差，越小越好
- **MAPE**: 平均绝对百分比误差，越小越好
- **方向准确率**: 预测涨跌方向的准确率，越高越好

## 🎨 自定义配置

### 修改预测参数
编辑 `config.py` 文件：
```python
DATA_CONFIG = {
    'sequence_length': 60,  # 输入序列长度
    'prediction_days': 5,   # 预测天数
}

MODEL_CONFIG = {
    'hidden_size': 128,     # 模型复杂度
    'num_layers': 3,        # 网络层数
}

TRAINING_CONFIG = {
    'num_epochs': 100,      # 训练轮数
    'batch_size': 32,       # 批次大小
}
```

### 添加新股票
只需要提供6位股票代码即可，系统会自动获取数据。

### 修改技术指标
在 `data_preprocessor.py` 中的 `select_features` 方法中添加或删除指标。

## 🔍 高级用法

### 批量预测
```python
from main import quick_predict

stocks = ['000001', '000002', '600036']
for stock in stocks:
    result = quick_predict(stock, days=3)
    print(f"{stock}: {result}")
```

### 模型比较
```bash
# 比较不同模型性能
python main.py --stock_code 000001 --model_type lstm --mode train
python main.py --stock_code 000001 --model_type gru --mode train
python main.py --stock_code 000001 --model_type transformer --mode train
```

### 自定义时间范围
```bash
python main.py --stock_code 000001 --start_date 2020-01-01 --end_date 2023-12-31
```

## 📁 文件结构说明

```
├── main.py              # 主程序
├── config.py            # 配置文件
├── data_fetcher.py      # 数据获取
├── data_preprocessor.py # 数据预处理
├── model.py             # 模型定义
├── trainer.py           # 训练模块
├── predictor.py         # 预测模块
├── visualizer.py        # 可视化
├── utils.py             # 工具函数
├── example.py           # 使用示例
├── quick_test.py        # 快速测试
├── install_dependencies.py # 智能安装
└── fix_network_issues.py   # 网络问题修复
```

## 🆘 获取帮助

### 命令行帮助
```bash
python main.py --help
```

### 常见问题
1. **Q: 预测准确吗？**
   A: 这是一个学习项目，预测结果仅供参考，不构成投资建议。

2. **Q: 可以预测多少天？**
   A: 建议1-7天，时间越长准确性越低。

3. **Q: 支持哪些股票？**
   A: 支持所有A股股票，提供6位代码即可。

4. **Q: 需要GPU吗？**
   A: 不需要，CPU也可以运行，GPU会更快。

### 联系支持
如果遇到问题，请：
1. 先运行 `python quick_test.py` 诊断
2. 查看错误日志
3. 检查网络连接
4. 尝试使用示例数据

## 🎉 开始使用

现在你可以开始使用A股股票预测系统了！

```bash
# 第一次使用推荐命令
python quick_test.py                    # 测试系统
python main.py --stock_code 000001 --mode both --days 3  # 开始预测
```

祝你使用愉快！ 📈🚀
