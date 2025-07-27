"""
智能依赖安装脚本
自动检测环境并安装合适的依赖包
"""

import sys
import subprocess
import platform
import os


def run_command(command):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_colab():
    """检查是否在Google Colab环境"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def install_package(package_name, alternative_names=None):
    """
    安装包，支持多个备选名称
    
    Args:
        package_name: 主要包名
        alternative_names: 备选包名列表
    """
    print(f"📦 尝试安装 {package_name}...")
    
    # 尝试安装主要包
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package_name}")
    
    if success:
        print(f"✅ {package_name} 安装成功")
        return True
    
    # 如果主要包安装失败，尝试备选包
    if alternative_names:
        for alt_name in alternative_names:
            print(f"🔄 尝试备选包: {alt_name}")
            success, stdout, stderr = run_command(f"{sys.executable} -m pip install {alt_name}")
            
            if success:
                print(f"✅ {alt_name} 安装成功")
                return True
            else:
                print(f"❌ {alt_name} 安装失败: {stderr}")
    
    print(f"❌ {package_name} 及其备选包都安装失败")
    return False


def install_talib():
    """智能安装TA-Lib库"""
    print("\n🔧 安装技术指标库...")
    
    system = platform.system().lower()
    is_colab = check_colab()
    
    if is_colab:
        print("🌟 检测到Google Colab环境")
        # 在Colab中，优先尝试ta库
        if install_package("ta"):
            return True
        # 如果ta库失败，尝试其他方法
        print("🔄 尝试安装TA-Lib...")
        success, _, _ = run_command("apt-get update && apt-get install -y libta-dev")
        if success:
            return install_package("TA-Lib")
        return False
    
    elif system == "windows":
        print("🪟 检测到Windows环境")
        # Windows环境下的安装顺序
        alternatives = [
            "talib-binary",  # 预编译二进制版本
            "TA-Lib",        # 官方版本
            "ta"             # 纯Python实现
        ]
        
        for package in alternatives:
            if install_package(package):
                return True
        return False
    
    elif system == "darwin":  # macOS
        print("🍎 检测到macOS环境")
        # 先尝试通过brew安装依赖
        print("📦 尝试通过Homebrew安装依赖...")
        run_command("brew install ta-lib")
        
        if install_package("TA-Lib"):
            return True
        return install_package("ta")
    
    else:  # Linux
        print("🐧 检测到Linux环境")
        # 先尝试安装系统依赖
        print("📦 尝试安装系统依赖...")
        run_command("sudo apt-get update")
        run_command("sudo apt-get install -y libta-dev")
        
        if install_package("TA-Lib"):
            return True
        return install_package("ta")


def main():
    """主安装函数"""
    print("🚀 智能依赖安装脚本")
    print("=" * 50)
    
    # 基础包列表
    basic_packages = [
        "torch",
        "torchvision", 
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "seaborn",
        "scikit-learn",
        "akshare",
        "tqdm",
        "joblib"
    ]
    
    # 安装基础包
    print("\n📦 安装基础依赖包...")
    failed_packages = []
    
    for package in basic_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    # 安装技术指标库
    talib_success = install_talib()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 安装结果汇总")
    print("=" * 50)
    
    if failed_packages:
        print("❌ 以下包安装失败:")
        for package in failed_packages:
            print(f"  - {package}")
    else:
        print("✅ 所有基础包安装成功")
    
    if talib_success:
        print("✅ 技术指标库安装成功")
    else:
        print("⚠️ 技术指标库安装失败，将使用简化版本")
    
    # 测试导入
    print("\n🧪 测试包导入...")
    test_imports()
    
    print("\n🎉 依赖安装完成！")
    
    if failed_packages or not talib_success:
        print("\n💡 如果遇到问题，请尝试:")
        print("1. 更新pip: python -m pip install --upgrade pip")
        print("2. 使用清华源: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/")
        print("3. 手动安装失败的包")


def test_imports():
    """测试关键包的导入"""
    test_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'akshare': 'AKShare',
        'sklearn': 'Scikit-learn'
    }
    
    for package, name in test_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} 导入成功")
        except ImportError:
            print(f"❌ {name} 导入失败")
    
    # 测试技术指标库
    talib_available = False
    ta_available = False
    
    try:
        import talib
        talib_available = True
        print("✅ TA-Lib 导入成功")
    except ImportError:
        try:
            import ta
            ta_available = True
            print("✅ ta 库导入成功")
        except ImportError:
            print("⚠️ 技术指标库导入失败，将使用简化版本")
    
    return talib_available or ta_available


if __name__ == "__main__":
    main()
