# Google Colab 安装指南

## 🚨 解决 talib-binary 安装问题

如果你遇到 `talib-binary` 安装错误，请按照以下步骤解决：

## 🔧 方法一：使用智能安装脚本（推荐）

在Colab中运行以下代码：

```python
# 1. 上传 install_dependencies.py 文件到Colab
from google.colab import files
uploaded = files.upload()

# 2. 运行智能安装脚本
!python install_dependencies.py
```

## 🔧 方法二：手动安装（逐步执行）

### 步骤1：安装基础依赖

```python
# 安装PyTorch和基础科学计算库
!pip install torch torchvision numpy pandas matplotlib plotly seaborn scikit-learn tqdm joblib

# 安装股票数据获取库
!pip install akshare
```

### 步骤2：安装技术指标库

选择以下任一方法：

#### 选项A：安装 ta 库（推荐，纯Python实现）
```python
!pip install ta
```

#### 选项B：在Colab中安装 TA-Lib
```python
# 安装系统依赖
!apt-get update
!apt-get install -y libta-dev

# 安装Python包
!pip install TA-Lib
```

#### 选项C：如果都失败，使用简化版本
```python
# 不安装任何技术指标库，系统会自动使用简化版本
print("将使用内置的简化技术指标计算")
```

### 步骤3：验证安装

```python
# 测试导入
def test_installation():
    try:
        import torch
        print("✅ PyTorch 可用")
    except ImportError:
        print("❌ PyTorch 不可用")
    
    try:
        import akshare
        print("✅ AKShare 可用")
    except ImportError:
        print("❌ AKShare 不可用")
    
    # 测试技术指标库
    talib_available = False
    ta_available = False
    
    try:
        import talib
        talib_available = True
        print("✅ TA-Lib 可用")
    except ImportError:
        try:
            import ta
            ta_available = True
            print("✅ ta 库可用")
        except ImportError:
            print("⚠️ 将使用简化版技术指标")
    
    return talib_available or ta_available

test_installation()
```

## 🚀 快速开始代码

安装完成后，使用以下代码快速开始：

```python
# 1. 上传项目文件
from google.colab import files
import zipfile

# 上传包含所有.py文件的zip包
uploaded = files.upload()

# 解压
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        break

# 2. 检查GPU
import torch
print(f"GPU可用: {torch.cuda.is_available()}")

# 3. 快速预测
from main_colab import colab_quick_predict

# 预测平安银行未来3天走势
result = colab_quick_predict('000001', days=3, mode='quick')
```

## 🛠️ 常见问题解决

### 问题1：talib-binary 找不到版本
**解决方案**：使用 `ta` 库替代
```python
!pip install ta
```

### 问题2：内存不足
**解决方案**：使用快速模式
```python
result = colab_quick_predict('000001', days=3, mode='quick')
```

### 问题3：会话超时
**解决方案**：
1. 及时保存结果
2. 使用快速模式减少训练时间
3. 分批处理多只股票

### 问题4：网络连接问题
**解决方案**：
```python
# 使用国内镜像源
!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ akshare
```

### 问题5：CUDA内存不足
**解决方案**：
```python
# 清理GPU内存
import torch
torch.cuda.empty_cache()

# 或者使用CPU模式
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## 📋 完整的Colab设置代码

将以下代码复制到Colab单元格中一次性执行：

```python
# === Google Colab 股票预测系统一键设置 ===

import subprocess
import sys
import warnings
warnings.filterwarnings('ignore')

print("🚀 开始设置Google Colab环境...")

# 1. 检查GPU
import torch
print(f"🔥 GPU可用: {torch.cuda.is_available()}")

# 2. 安装基础依赖
print("📦 安装基础依赖...")
basic_packages = [
    'akshare', 'plotly', 'seaborn', 'tqdm', 'joblib', 'scikit-learn'
]

for package in basic_packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        print(f"✅ {package}")
    except:
        print(f"❌ {package}")

# 3. 安装技术指标库
print("🔧 安装技术指标库...")
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ta'])
    print("✅ ta 库安装成功")
except:
    print("⚠️ 技术指标库安装失败，将使用简化版本")

# 4. 创建目录
import os
for directory in ['data', 'models', 'results']:
    os.makedirs(directory, exist_ok=True)

print("🎉 环境设置完成！")
print("💡 现在请上传项目文件的zip包")
```

## 📱 使用专用Notebook

最简单的方法是直接使用我们提供的 `Stock_Prediction_Colab.ipynb` 文件：

1. 下载 `Stock_Prediction_Colab.ipynb`
2. 上传到Google Colab
3. 按照notebook中的步骤执行

这个notebook已经包含了所有必要的安装和配置代码，可以避免大部分安装问题。

## 🆘 如果仍有问题

如果按照上述方法仍然遇到问题，请：

1. **重启运行时**：运行时 → 重启运行时
2. **检查网络**：确保网络连接正常
3. **使用简化版本**：系统支持不安装技术指标库的简化模式
4. **联系支持**：提供具体的错误信息

记住：即使技术指标库安装失败，系统仍然可以正常运行，只是会使用简化版的技术指标计算。
