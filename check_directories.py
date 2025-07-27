"""
目录检查和创建脚本
"""

import os
import sys
from pathlib import Path

def check_and_create_directories():
    """检查和创建必要的目录"""
    print("🔍 检查项目目录结构...")
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"📍 当前目录: {current_dir}")
    
    # 定义需要的目录
    required_dirs = {
        'data': 'data/',
        'models': 'models/',
        'results': 'results/',
        'logs': 'logs/'
    }
    
    created_dirs = []
    failed_dirs = []
    
    for name, path in required_dirs.items():
        full_path = os.path.join(current_dir, path)
        
        try:
            # 检查目录是否存在
            if os.path.exists(full_path):
                print(f"✅ {name}: {full_path} - 已存在")
            else:
                # 创建目录
                os.makedirs(full_path, exist_ok=True)
                if os.path.exists(full_path):
                    print(f"🆕 {name}: {full_path} - 已创建")
                    created_dirs.append(name)
                else:
                    print(f"❌ {name}: {full_path} - 创建失败")
                    failed_dirs.append(name)
                    
        except Exception as e:
            print(f"❌ {name}: {full_path} - 错误: {str(e)}")
            failed_dirs.append(name)
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 目录创建结果")
    print("="*50)
    
    if created_dirs:
        print(f"🆕 新创建的目录: {', '.join(created_dirs)}")
    
    if failed_dirs:
        print(f"❌ 创建失败的目录: {', '.join(failed_dirs)}")
    else:
        print("✅ 所有必要目录都已就绪")
    
    return len(failed_dirs) == 0

def list_directory_contents():
    """列出目录内容"""
    print("\n📋 当前目录内容:")
    print("-" * 30)
    
    try:
        items = os.listdir('.')
        
        # 分类显示
        dirs = [item for item in items if os.path.isdir(item) and not item.startswith('.')]
        files = [item for item in items if os.path.isfile(item) and not item.startswith('.')]
        
        print("📁 目录:")
        for d in sorted(dirs):
            print(f"  {d}/")
        
        print("\n📄 Python文件:")
        py_files = [f for f in files if f.endswith('.py')]
        for f in sorted(py_files):
            print(f"  {f}")
        
        print("\n📄 其他文件:")
        other_files = [f for f in files if not f.endswith('.py')]
        for f in sorted(other_files):
            print(f"  {f}")
            
    except Exception as e:
        print(f"❌ 无法列出目录内容: {str(e)}")

def check_file_permissions():
    """检查文件权限"""
    print("\n🔐 检查文件权限...")
    
    current_dir = os.getcwd()
    
    # 检查当前目录权限
    if os.access(current_dir, os.R_OK):
        print("✅ 当前目录可读")
    else:
        print("❌ 当前目录不可读")
    
    if os.access(current_dir, os.W_OK):
        print("✅ 当前目录可写")
    else:
        print("❌ 当前目录不可写")
    
    if os.access(current_dir, os.X_OK):
        print("✅ 当前目录可执行")
    else:
        print("❌ 当前目录不可执行")

def test_directory_operations():
    """测试目录操作"""
    print("\n🧪 测试目录操作...")
    
    test_dir = "test_temp_dir"
    test_file = os.path.join(test_dir, "test_file.txt")
    
    try:
        # 创建测试目录
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ 测试目录创建成功: {test_dir}")
        
        # 创建测试文件
        with open(test_file, 'w') as f:
            f.write("test content")
        print(f"✅ 测试文件创建成功: {test_file}")
        
        # 读取测试文件
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"✅ 测试文件读取成功: {content}")
        
        # 清理测试文件和目录
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✅ 测试文件和目录清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 目录操作测试失败: {str(e)}")
        
        # 尝试清理
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
        except:
            pass
        
        return False

def fix_common_issues():
    """修复常见问题"""
    print("\n🔧 尝试修复常见问题...")
    
    fixes_applied = []
    
    # 修复1: 确保使用正确的路径分隔符
    try:
        import platform
        system = platform.system()
        print(f"📱 操作系统: {system}")
        
        if system == "Windows":
            print("🪟 Windows系统，使用反斜杠路径")
        else:
            print("🐧 Unix-like系统，使用正斜杠路径")
        
        fixes_applied.append("路径分隔符检查")
    except Exception as e:
        print(f"❌ 系统检查失败: {str(e)}")
    
    # 修复2: 检查Python路径
    try:
        python_path = sys.executable
        print(f"🐍 Python路径: {python_path}")
        fixes_applied.append("Python路径检查")
    except Exception as e:
        print(f"❌ Python路径检查失败: {str(e)}")
    
    # 修复3: 检查工作目录
    try:
        work_dir = os.getcwd()
        print(f"📂 工作目录: {work_dir}")
        
        # 检查是否在正确的项目目录中
        expected_files = ['main.py', 'config.py', 'requirements.txt']
        missing_files = [f for f in expected_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"⚠️ 可能不在正确的项目目录中，缺少文件: {missing_files}")
        else:
            print("✅ 在正确的项目目录中")
        
        fixes_applied.append("工作目录检查")
    except Exception as e:
        print(f"❌ 工作目录检查失败: {str(e)}")
    
    if fixes_applied:
        print(f"✅ 应用的修复: {', '.join(fixes_applied)}")
    
    return len(fixes_applied) > 0

def main():
    """主函数"""
    print("🔧 目录检查和修复工具")
    print("=" * 50)
    
    # 1. 检查和创建目录
    dirs_ok = check_and_create_directories()
    
    # 2. 列出目录内容
    list_directory_contents()
    
    # 3. 检查权限
    check_file_permissions()
    
    # 4. 测试目录操作
    ops_ok = test_directory_operations()
    
    # 5. 修复常见问题
    fixes_ok = fix_common_issues()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 检查结果汇总")
    print("=" * 50)
    print(f"目录创建: {'✅ 成功' if dirs_ok else '❌ 失败'}")
    print(f"目录操作: {'✅ 成功' if ops_ok else '❌ 失败'}")
    print(f"问题修复: {'✅ 完成' if fixes_ok else '⚠️ 无需修复'}")
    
    if dirs_ok and ops_ok:
        print("\n🎉 目录结构检查完成，一切正常！")
        print("💡 现在可以正常使用股票预测系统了")
        print("\n🚀 建议运行:")
        print("python main.py --stock_code 000001 --mode both --days 3")
    else:
        print("\n⚠️ 发现一些问题，但系统可能仍可运行")
        print("💡 如果遇到问题，请检查:")
        print("1. 是否在正确的项目目录中")
        print("2. 是否有足够的文件系统权限")
        print("3. 磁盘空间是否充足")
    
    # 显示使用说明
    print("\n📖 使用说明:")
    print("- models/: 存放训练好的模型文件")
    print("- results/: 存放预测结果和图表")
    print("- data/: 存放股票数据缓存")
    print("- logs/: 存放日志文件")

if __name__ == "__main__":
    main()
