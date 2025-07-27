# A股股票预测深度学习系统

这是一个基于深度学习的中国A股股票K线图走势预测系统，支持LSTM、GRU和Transformer等多种模型架构。

## 🚀 功能特点

- **多模型支持**: LSTM、GRU、Transformer三种深度学习模型
- **丰富的技术指标**: 包含20+种技术指标（MA、MACD、RSI、KDJ、布林带等）
- **完整的数据流程**: 数据获取、预处理、训练、预测、可视化一体化
- **实时数据获取**: 使用akshare库获取最新的A股数据
- **交互式可视化**: 支持K线图、技术指标图、预测结果图等多种可视化
- **模型评估**: 提供RMSE、MAE、MAPE、方向准确率等多种评估指标
- **Google Colab支持**: 免费GPU训练，无需本地显卡资源

## 🌟 Google Colab 运行教程

### 方法一：直接上传文件运行

#### 1. 准备文件
将以下文件打包成zip文件：
- `main.py`
- `config.py`
- `data_fetcher.py`
- `data_preprocessor.py`
- `model.py`
- `trainer.py`
- `predictor.py`
- `visualizer.py`
- `utils.py`
- `requirements.txt`

#### 2. 在Colab中运行
打开 [Google Colab](https://colab.research.google.com/)，创建新的notebook，然后执行以下代码：

```python
# 1. 上传项目文件
from google.colab import files
import zipfile
import os

# 上传zip文件
uploaded = files.upload()

# 解压文件
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        break

# 查看文件
!ls -la
```

```python
# 2. 安装依赖
!pip install torch torchvision
!pip install akshare
!pip install talib-binary
!pip install plotly
!pip install seaborn
!pip install tqdm
!pip install joblib
```

```python
# 3. 检查GPU可用性
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

```python
# 4. 运行系统测试
!python test_system.py
```

```python
# 5. 开始预测（以平安银行为例）
!python main.py --stock_code 000001 --mode both --days 5 --model_type lstm
```

### 方法二：从GitHub克隆运行

如果你将代码上传到GitHub，可以直接克隆：

```python
# 1. 克隆仓库
!git clone https://github.com/your-username/stock-prediction.git
%cd stock-prediction

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 运行预测
!python main.py --stock_code 000001 --mode both --days 5
```

### 方法三：逐步运行（推荐用于学习）

```python
# 1. 安装依赖
!pip install torch torchvision akshare talib-binary plotly seaborn tqdm joblib scikit-learn

# 2. 创建项目文件（将下面的代码分别保存为对应的.py文件）
# 然后逐个运行各个模块
```

### Colab专用配置调整

在Colab中运行时，建议调整以下配置以适应免费GPU的限制：

```python
# 修改config.py中的参数
DATA_CONFIG = {
    'sequence_length': 30,  # 减少序列长度以节省内存
    'prediction_days': 3,   # 减少预测天数
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

MODEL_CONFIG = {
    'input_size': 20,
    'hidden_size': 64,      # 减少隐藏层大小
    'num_layers': 2,        # 减少层数
    'dropout': 0.2,
    'bidirectional': True,
}

TRAINING_CONFIG = {
    'batch_size': 16,       # 减少批次大小
    'learning_rate': 0.001,
    'num_epochs': 50,       # 减少训练轮数
    'patience': 8,
    'weight_decay': 1e-5,
}
```

### Colab运行示例

```python
# 完整的Colab运行示例
import warnings
warnings.filterwarnings('ignore')

# 快速预测示例
from main import quick_predict

# 预测平安银行未来3天走势
result = quick_predict('000001', days=3)

if result:
    print("\n预测结果:")
    for i, (date, price) in enumerate(zip(result['dates'], result['predictions'])):
        change = price - result['last_price']
        change_pct = change / result['last_price'] * 100
        print(f"第{i+1}天 ({date.strftime('%Y-%m-%d')}): "
              f"{price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
```

### 方法四：使用专用Colab Notebook（推荐）

我们提供了专门的Colab Notebook文件 `Stock_Prediction_Colab.ipynb`：

1. 下载 `Stock_Prediction_Colab.ipynb` 文件
2. 上传到Google Colab
3. 按照notebook中的步骤执行

### 方法五：使用Colab专用程序

```python
# 使用Colab优化版本
from main_colab import colab_quick_predict

# 快速预测
result = colab_quick_predict('000001', days=3, model_type='lstm', mode='quick')

# 批量预测
from main_colab import colab_batch_predict
results = colab_batch_predict(['000001', '000002', '600036'], days=3)
```

### Colab运行模式

系统提供三种运行模式：

1. **快速模式** (`mode='quick'`): 适合演示和测试
   - 序列长度: 20天
   - 训练轮数: 10轮
   - 隐藏层: 32

2. **标准模式** (`mode='normal'`): 平衡速度和精度
   - 序列长度: 30天
   - 训练轮数: 30轮
   - 隐藏层: 64

3. **性能模式** (`mode='performance'`): 最佳精度（需要更多资源）
   - 序列长度: 60天
   - 训练轮数: 100轮
   - 隐藏层: 128

### 注意事项

1. **GPU使用限制**: Colab免费版每天有GPU使用时间限制
2. **会话超时**: 长时间不活动会断开连接，建议分段运行
3. **文件保存**: 训练好的模型会在会话结束后丢失，建议下载保存
4. **内存限制**: 如遇内存不足，请减少批次大小和模型复杂度
5. **网络问题**: 数据获取依赖网络，如失败请重试

### 保存和下载结果

```python
# 下载训练好的模型和结果
from google.colab import files
import shutil

# 打包结果文件
!zip -r results.zip models/ results/ data/

# 下载
files.download('results.zip')
```

### Colab性能优化建议

1. **启用GPU**: 运行时 → 更改运行时类型 → GPU
2. **使用快速模式**: 首次运行建议使用快速模式
3. **分批处理**: 批量预测时建议每次处理3-5只股票
4. **及时保存**: 重要结果及时下载保存

## 📦 安装依赖

### 智能安装（推荐）

```bash
# 使用智能安装脚本，自动处理不同环境的兼容性问题
python install_dependencies.py
```

### 手动安装

如果遇到 `talib-binary` 安装问题，请按以下顺序尝试：

#### 方法1：使用 ta 库（推荐）
```bash
pip install torch numpy pandas matplotlib plotly seaborn scikit-learn akshare tqdm joblib
pip install ta  # 纯Python实现，兼容性最好
```

#### 方法2：Windows用户
```bash
pip install talib-binary  # Windows预编译版本
```

#### 方法3：macOS用户
```bash
brew install ta-lib  # 先安装系统依赖
pip install TA-Lib
```

#### 方法4：Linux用户
```bash
sudo apt-get install libta-dev  # Ubuntu/Debian
# 或
sudo yum install ta-lib-devel   # CentOS/RHEL
pip install TA-Lib
```

#### 方法5：使用requirements.txt
```bash
pip install -r requirements.txt
```

### 🚨 常见问题解决

#### 1. 安装问题
如果遇到安装问题，请查看 [COLAB_INSTALL_GUIDE.md](COLAB_INSTALL_GUIDE.md) 获取详细的解决方案。

#### 2. 网络问题
如果遇到股票数据获取失败：

```bash
# 运行网络问题诊断脚本
python fix_network_issues.py

# 或运行快速测试
python quick_test.py
```

#### 3. 离线模式
如果网络不稳定，系统支持离线模式：
- 系统会自动创建示例数据
- 可以正常进行模型训练和测试
- 所有功能都可以离线使用

**重要提示**：
- 即使技术指标库安装失败，系统仍然可以正常运行
- 即使无法获取真实股票数据，系统会使用示例数据
- 系统设计为高容错性，确保在各种环境下都能运行

## 🎯 快速开始

### 1. 基本使用

```bash
# 训练并预测平安银行(000001)未来5天走势
python main.py --stock_code 000001 --mode both --days 5

# 仅训练模型
python main.py --stock_code 000001 --mode train

# 仅进行预测（需要先训练模型）
python main.py --stock_code 000001 --mode predict --days 5
```

### 2. 指定模型类型

```bash
# 使用LSTM模型
python main.py --stock_code 000001 --model_type lstm

# 使用GRU模型
python main.py --stock_code 000001 --model_type gru

# 使用Transformer模型
python main.py --stock_code 000001 --model_type transformer
```

### 3. 自定义时间范围

```bash
# 指定数据时间范围
python main.py --stock_code 000001 --start_date 2020-01-01 --end_date 2023-12-31
```

## 📊 使用示例

运行示例程序：

```bash
python example.py
```

示例包含：
1. 快速预测
2. 分步骤详细使用
3. 比较不同模型性能
4. 批量预测多只股票

## 🏗️ 系统架构

```
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── data_fetcher.py      # 数据获取模块
├── data_preprocessor.py # 数据预处理模块
├── model.py             # 深度学习模型定义
├── trainer.py           # 模型训练模块
├── predictor.py         # 预测模块
├── visualizer.py        # 可视化模块
├── utils.py             # 工具函数
├── example.py           # 使用示例
└── requirements.txt     # 依赖包列表
```

## 🔧 配置说明

在 `config.py` 中可以调整以下参数：

### 数据配置
- `sequence_length`: 输入序列长度（默认60天）
- `prediction_days`: 预测天数（默认5天）
- `train_ratio`: 训练集比例（默认0.8）

### 模型配置
- `hidden_size`: LSTM隐藏层大小（默认128）
- `num_layers`: LSTM层数（默认3）
- `dropout`: Dropout率（默认0.2）

### 训练配置
- `batch_size`: 批次大小（默认32）
- `learning_rate`: 学习率（默认0.001）
- `num_epochs`: 训练轮数（默认100）

## 📈 支持的技术指标

- **移动平均线**: MA5, MA10, MA20, MA60
- **指数移动平均**: EMA12, EMA26
- **MACD**: MACD线、信号线、柱状图
- **相对强弱指标**: RSI
- **布林带**: 上轨、中轨、下轨
- **随机指标**: KDJ
- **威廉指标**: WR
- **成交量指标**: 成交量比率
- **价格变化**: 涨跌幅、振幅等

## 🎨 可视化功能

系统提供多种可视化功能：

1. **K线图**: 交互式K线图，包含移动平均线
2. **技术指标图**: RSI、KDJ、布林带、成交量等
3. **训练历史**: 训练和验证损失曲线
4. **预测结果**: 实际值vs预测值对比
5. **未来预测**: 未来几天的价格预测图

## 📊 模型评估指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **方向准确率**: 预测涨跌方向的准确率

## 🔍 常用股票代码

- 000001: 平安银行
- 000002: 万科A
- 600036: 招商银行
- 600519: 贵州茅台
- 000858: 五粮液
- 002415: 海康威视

## ⚠️ 注意事项

1. **免责声明**: 本系统仅供学习和研究使用，不构成投资建议
2. **数据延迟**: 股票数据可能存在延迟，请以实际交易数据为准
3. **模型限制**: 深度学习模型无法保证预测准确性，投资有风险
4. **硬件要求**: 建议使用GPU加速训练，CPU训练速度较慢

## 🛠️ 故障排除

### 常见问题

1. **数据获取失败**
   - 检查网络连接
   - 确认股票代码格式正确
   - 尝试更换时间范围

2. **模型训练缓慢**
   - 减少训练轮数
   - 降低模型复杂度
   - 使用GPU加速

3. **内存不足**
   - 减少批次大小
   - 缩短序列长度
   - 减少特征数量

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持LSTM、GRU、Transformer模型
- 完整的数据处理和可视化功能
- 多种技术指标支持

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。
