"""
Google Colab环境快速设置脚本
一键安装依赖和配置环境
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def check_colab_environment():
    """检查是否在Colab环境中"""
    try:
        import google.colab
        print("🌟 检测到Google Colab环境")
        return True
    except ImportError:
        print("💻 本地环境")
        return False

def install_dependencies():
    """安装必要的依赖包"""
    print("📦 开始安装依赖包...")

    # 基础包列表（不包含可能有问题的包）
    basic_packages = [
        'akshare',
        'plotly',
        'seaborn',
        'tqdm',
        'joblib',
        'scikit-learn'
    ]

    failed_packages = []

    for package in basic_packages:
        try:
            print(f"📥 安装 {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            failed_packages.append(package)

    # 智能安装技术指标库
    print("\n🔧 安装技术指标库...")
    talib_success = install_talib_smart()

    if failed_packages:
        print(f"⚠️ {len(failed_packages)} 个包安装失败，但系统仍可运行")
        return False
    elif not talib_success:
        print("⚠️ 技术指标库安装失败，将使用简化版本")
        return True
    else:
        print("🎉 所有依赖包安装完成！")
        return True

def install_talib_smart():
    """智能安装技术指标库"""
    # 首先尝试ta库（纯Python实现，兼容性最好）
    try:
        print("📥 尝试安装 ta 库...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ta'])
        print("✅ ta 库安装成功")
        return True
    except subprocess.CalledProcessError:
        print("❌ ta 库安装失败")

    # 如果在Colab环境，尝试安装TA-Lib
    if check_colab_environment():
        try:
            print("📥 在Colab中尝试安装 TA-Lib...")
            # 先安装系统依赖
            subprocess.check_call(['apt-get', 'update'], stdout=subprocess.DEVNULL)
            subprocess.check_call(['apt-get', 'install', '-y', 'libta-dev'], stdout=subprocess.DEVNULL)
            # 再安装Python包
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'TA-Lib'])
            print("✅ TA-Lib 安装成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ TA-Lib 安装失败")

    print("⚠️ 技术指标库安装失败，将使用简化版本")
    return False

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def create_project_structure():
    """创建项目目录结构"""
    print("📁 创建项目目录结构...")
    
    directories = ['data', 'models', 'results']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📂 创建目录: {directory}/")
    
    print("✅ 项目目录结构创建完成")

def download_sample_data():
    """下载示例数据进行测试"""
    print("📊 测试数据获取功能...")
    
    try:
        import akshare as ak
        
        # 测试获取股票数据
        print("🔍 测试获取平安银行(000001)数据...")
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20230101", end_date="20231231", adjust="qfq")
        
        if df is not None and len(df) > 0:
            print(f"✅ 数据获取成功，共 {len(df)} 条记录")
            
            # 保存示例数据
            df.to_csv('data/000001_sample.csv')
            print("💾 示例数据已保存到 data/000001_sample.csv")
            return True
        else:
            print("❌ 数据获取失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据获取测试失败: {str(e)}")
        return False

def create_quick_start_script():
    """创建快速开始脚本"""
    print("📝 创建快速开始脚本...")
    
    script_content = '''
# 快速开始脚本
# 复制以下代码到新的Colab单元格中运行

# 1. 导入必要的库
import warnings
warnings.filterwarnings('ignore')

# 2. 快速预测示例
def quick_demo():
    """快速演示"""
    print("🚀 开始快速演示...")
    
    # 这里需要你上传项目文件后才能运行
    try:
        from main_colab import colab_quick_predict
        
        # 预测平安银行未来3天走势
        result = colab_quick_predict('000001', days=3, model_type='lstm', mode='quick')
        
        if result:
            print("🎉 演示成功完成！")
            return result
        else:
            print("❌ 演示失败")
            return None
            
    except ImportError:
        print("❌ 请先上传项目文件")
        print("💡 提示：将所有.py文件打包成zip上传到Colab")
        return None

# 运行演示
# quick_demo()
'''
    
    with open('quick_start.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ 快速开始脚本已创建: quick_start.py")

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("📖 使用说明")
    print("="*60)
    
    instructions = """
🎯 接下来的步骤：

1. 📁 上传项目文件
   - 将所有.py文件打包成zip文件
   - 在Colab中上传并解压

2. 🚀 开始预测
   ```python
   from main_colab import colab_quick_predict
   result = colab_quick_predict('000001', days=3, mode='quick')
   ```

3. 📊 批量预测
   ```python
   from main_colab import colab_batch_predict
   results = colab_batch_predict(['000001', '000002'], days=3)
   ```

4. 💾 保存结果
   ```python
   from google.colab import files
   !zip -r results.zip models/ results/
   files.download('results.zip')
   ```

🔧 常用股票代码：
   000001: 平安银行    000002: 万科A
   600036: 招商银行    600519: 贵州茅台
   000858: 五粮液      002415: 海康威视

⚠️ 注意事项：
   - 确保已启用GPU加速
   - 首次运行建议使用快速模式
   - 及时保存重要结果
   - 遇到内存不足请减少批次大小

🆘 如遇问题：
   - 检查网络连接
   - 重启运行时
   - 减少数据量和模型复杂度
"""
    
    print(instructions)

def main():
    """主函数"""
    print("🚀 Google Colab环境设置脚本")
    print("="*50)
    
    # 检查环境
    is_colab = check_colab_environment()
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败，请检查网络连接")
        return
    
    # 检查GPU
    gpu_available = check_gpu()
    if not gpu_available:
        print("💡 建议启用GPU：运行时 → 更改运行时类型 → GPU")
    
    # 创建项目结构
    create_project_structure()
    
    # 测试数据获取
    data_ok = download_sample_data()
    if not data_ok:
        print("⚠️ 数据获取测试失败，可能是网络问题")
    
    # 创建快速开始脚本
    create_quick_start_script()
    
    # 打印使用说明
    print_usage_instructions()
    
    print("\n🎉 环境设置完成！")
    print("💡 现在可以上传项目文件并开始使用了")

if __name__ == "__main__":
    main()
