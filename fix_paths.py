"""
修复路径问题脚本
检查和修复模型、结果保存路径问题
"""

import os
import sys
import shutil
from pathlib import Path

def check_current_directory():
    """检查当前工作目录"""
    print("🔍 检查当前工作目录...")
    
    current_dir = os.getcwd()
    print(f"📍 当前目录: {current_dir}")
    
    # 检查是否在正确的项目目录
    expected_files = ['main.py', 'config.py', 'trainer.py', 'predictor.py']
    missing_files = [f for f in expected_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️ 可能不在正确的项目目录中，缺少文件: {missing_files}")
        return False
    else:
        print("✅ 在正确的项目目录中")
        return True

def check_directories():
    """检查目录结构"""
    print("\n📁 检查目录结构...")
    
    required_dirs = ['data', 'models', 'results', 'logs']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ - 存在")
        else:
            print(f"❌ {dir_name}/ - 不存在，正在创建...")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"🆕 {dir_name}/ - 已创建")
            except Exception as e:
                print(f"❌ 创建失败: {str(e)}")

def find_model_files():
    """查找模型文件"""
    print("\n🔍 查找模型文件...")
    
    # 在当前目录及子目录中查找.pth文件
    model_files = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                model_files.append(full_path)
    
    if model_files:
        print("📦 找到的模型文件:")
        for file in model_files:
            print(f"  {file}")
        return model_files
    else:
        print("❌ 未找到模型文件")
        return []

def find_result_files():
    """查找结果文件"""
    print("\n🔍 查找结果文件...")
    
    # 查找图片和JSON文件
    result_extensions = ['.png', '.jpg', '.json', '.csv']
    result_files = []
    
    for root, dirs, files in os.walk('.'):
        # 跳过系统目录
        if any(skip in root for skip in ['.git', '__pycache__', '.ipynb_checkpoints']):
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in result_extensions):
                # 只包含可能是结果文件的
                if any(keyword in file.lower() for keyword in ['prediction', 'model', 'stock', '000001', 'result', 'chart']):
                    full_path = os.path.join(root, file)
                    result_files.append(full_path)
    
    if result_files:
        print("📊 找到的结果文件:")
        for file in result_files:
            print(f"  {file}")
        return result_files
    else:
        print("❌ 未找到结果文件")
        return []

def move_files_to_correct_directories():
    """移动文件到正确的目录"""
    print("\n🔄 移动文件到正确目录...")
    
    moved_count = 0
    
    # 移动模型文件
    model_files = find_model_files()
    for file_path in model_files:
        if not file_path.startswith('./models/'):
            filename = os.path.basename(file_path)
            target_path = os.path.join('models', filename)
            
            try:
                shutil.move(file_path, target_path)
                print(f"📦 移动模型: {file_path} -> {target_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败: {file_path} - {str(e)}")
    
    # 移动结果文件
    result_files = find_result_files()
    for file_path in result_files:
        if not file_path.startswith('./results/'):
            filename = os.path.basename(file_path)
            target_path = os.path.join('results', filename)
            
            try:
                # 避免覆盖同名文件
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_path):
                        new_filename = f"{base}_{counter}{ext}"
                        target_path = os.path.join('results', new_filename)
                        counter += 1
                
                shutil.move(file_path, target_path)
                print(f"📊 移动结果: {file_path} -> {target_path}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 移动失败: {file_path} - {str(e)}")
    
    print(f"\n✅ 共移动了 {moved_count} 个文件")

def check_config_paths():
    """检查配置文件中的路径"""
    print("\n🔧 检查配置文件路径...")
    
    try:
        from config import PATHS
        
        print("📋 配置的路径:")
        for key, path in PATHS.items():
            abs_path = os.path.abspath(path)
            exists = os.path.exists(path)
            print(f"  {key}: {path} -> {abs_path} ({'✅' if exists else '❌'})")
        
        return True
    except Exception as e:
        print(f"❌ 无法读取配置文件: {str(e)}")
        return False

def test_file_operations():
    """测试文件操作"""
    print("\n🧪 测试文件操作...")
    
    test_files = {
        'models/test_model.txt': '这是一个测试模型文件',
        'results/test_result.txt': '这是一个测试结果文件',
        'data/test_data.txt': '这是一个测试数据文件'
    }
    
    success_count = 0
    
    for file_path, content in test_files.items():
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 写入测试文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 验证文件存在
            if os.path.exists(file_path):
                print(f"✅ 测试文件创建成功: {file_path}")
                success_count += 1
                
                # 清理测试文件
                os.remove(file_path)
            else:
                print(f"❌ 测试文件创建失败: {file_path}")
                
        except Exception as e:
            print(f"❌ 测试失败 {file_path}: {str(e)}")
    
    print(f"📊 文件操作测试: {success_count}/{len(test_files)} 成功")
    return success_count == len(test_files)

def update_config_with_absolute_paths():
    """更新配置文件使用绝对路径"""
    print("\n🔧 更新配置文件...")
    
    try:
        current_dir = os.getcwd()
        
        # 创建新的配置内容
        new_config = f'''"""
配置文件 - 使用绝对路径
"""

# 数据配置
DATA_CONFIG = {{
    'sequence_length': 60,  # 输入序列长度
    'prediction_days': 5,   # 预测天数
    'train_ratio': 0.8,     # 训练集比例
    'val_ratio': 0.1,       # 验证集比例
    'test_ratio': 0.1,      # 测试集比例
}}

# 模型配置
MODEL_CONFIG = {{
    'input_size': 20,       # 输入特征数量
    'hidden_size': 128,     # 隐藏层大小
    'num_layers': 3,        # LSTM层数
    'dropout': 0.2,         # Dropout率
    'bidirectional': True,  # 是否使用双向LSTM
}}

# 训练配置
TRAINING_CONFIG = {{
    'batch_size': 32,       # 批次大小
    'learning_rate': 0.001, # 学习率
    'num_epochs': 100,      # 训练轮数
    'patience': 10,         # 早停耐心值
    'weight_decay': 1e-5,   # L2正则化
}}

# 数据路径 - 使用绝对路径
PATHS = {{
    'data_dir': r'{os.path.join(current_dir, "data")}',
    'model_dir': r'{os.path.join(current_dir, "models")}',
    'results_dir': r'{os.path.join(current_dir, "results")}',
}}

# 股票代码示例
DEFAULT_STOCK_CODE = '000001'  # 平安银行
'''
        
        # 备份原配置文件
        if os.path.exists('config.py'):
            shutil.copy('config.py', 'config_backup.py')
            print("📋 原配置文件已备份为 config_backup.py")
        
        # 写入新配置
        with open('config_absolute_paths.py', 'w', encoding='utf-8') as f:
            f.write(new_config)
        
        print("✅ 创建了使用绝对路径的配置文件: config_absolute_paths.py")
        print("💡 如果问题持续存在，可以将此文件重命名为 config.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 更新配置文件失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔧 路径问题修复工具")
    print("=" * 50)
    
    # 1. 检查当前目录
    dir_ok = check_current_directory()
    
    # 2. 检查目录结构
    check_directories()
    
    # 3. 查找现有文件
    model_files = find_model_files()
    result_files = find_result_files()
    
    # 4. 移动文件到正确位置
    if model_files or result_files:
        move_files_to_correct_directories()
    
    # 5. 检查配置路径
    config_ok = check_config_paths()
    
    # 6. 测试文件操作
    test_ok = test_file_operations()
    
    # 7. 如果有问题，创建绝对路径配置
    if not test_ok:
        update_config_with_absolute_paths()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 修复结果汇总")
    print("=" * 50)
    print(f"目录检查: {'✅ 正常' if dir_ok else '⚠️ 异常'}")
    print(f"配置检查: {'✅ 正常' if config_ok else '❌ 异常'}")
    print(f"文件操作: {'✅ 正常' if test_ok else '❌ 异常'}")
    
    if dir_ok and config_ok and test_ok:
        print("\n🎉 路径问题已修复！")
        print("💡 现在模型和结果应该会正确保存到对应目录")
    else:
        print("\n⚠️ 仍存在一些问题，但已尽力修复")
        print("💡 建议:")
        print("1. 确保在正确的项目目录中运行程序")
        print("2. 检查文件系统权限")
        print("3. 如果问题持续，使用绝对路径配置")
    
    print("\n🚀 现在可以重新运行股票预测:")
    print("python main.py --stock_code 000001 --mode both --days 3")

if __name__ == "__main__":
    main()
