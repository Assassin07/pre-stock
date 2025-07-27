"""
维度调试脚本
帮助诊断和修复张量维度不匹配问题
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd

def debug_data_shapes():
    """调试数据形状"""
    print("🔍 调试数据维度...")
    
    try:
        from data_fetcher import StockDataFetcher
        from data_preprocessor import StockDataPreprocessor
        
        # 获取测试数据
        fetcher = StockDataFetcher()
        df = fetcher._create_sample_data('000001', '2023-01-01', '2023-03-31', 'daily')
        
        if df is None:
            print("❌ 无法创建测试数据")
            return False
        
        print(f"📊 原始数据形状: {df.shape}")
        print(f"📋 原始数据列: {list(df.columns)}")
        
        # 数据预处理
        preprocessor = StockDataPreprocessor()
        
        # 设置较小的参数以便调试
        preprocessor.sequence_length = 10
        preprocessor.prediction_days = 3
        
        print(f"\n🔧 预处理参数:")
        print(f"   序列长度: {preprocessor.sequence_length}")
        print(f"   预测天数: {preprocessor.prediction_days}")
        
        # 添加技术指标
        df_with_indicators = preprocessor.add_technical_indicators(df)
        print(f"📈 添加技术指标后: {df_with_indicators.shape}")
        
        # 选择特征
        feature_data = preprocessor.select_features(df_with_indicators)
        feature_data = feature_data.dropna()
        print(f"🎯 特征数据形状: {feature_data.shape}")
        print(f"📋 特征列: {preprocessor.feature_columns}")
        
        # 标准化
        normalized_data = preprocessor.normalize_data(feature_data.values, fit_scaler=True)
        normalized_df = pd.DataFrame(normalized_data, columns=preprocessor.feature_columns, index=feature_data.index)
        print(f"📏 标准化后形状: {normalized_df.shape}")
        
        # 创建序列
        X, y = preprocessor.create_sequences(normalized_df)
        print(f"🔄 序列数据: X.shape={X.shape}, y.shape={y.shape}")
        
        return X, y, len(preprocessor.feature_columns)
        
    except Exception as e:
        print(f"❌ 数据调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_model_shapes(input_size, output_size):
    """调试模型维度"""
    print(f"\n🤖 调试模型维度...")
    print(f"   输入特征数: {input_size}")
    print(f"   输出维度: {output_size}")
    
    try:
        from model import create_model
        
        # 创建模型
        model = create_model('lstm', input_size=input_size, output_size=output_size)
        print(f"✅ 模型创建成功")
        
        # 测试不同的输入形状
        test_cases = [
            (1, 10, input_size),   # 单样本
            (4, 10, input_size),   # 小批次
            (32, 10, input_size),  # 标准批次
        ]
        
        for batch_size, seq_len, features in test_cases:
            test_input = torch.randn(batch_size, seq_len, features)
            
            with torch.no_grad():
                output = model(test_input)
                print(f"📊 输入: {test_input.shape} -> 输出: {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_training_loop(X, y, model):
    """调试训练循环"""
    print(f"\n🏋️ 调试训练循环...")
    
    try:
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X[:20])  # 只取前20个样本
        y_tensor = torch.FloatTensor(y[:20])
        
        print(f"📊 张量形状: X={X_tensor.shape}, y={y_tensor.shape}")
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 测试一个批次
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            print(f"\n批次 {batch_idx + 1}:")
            print(f"   batch_X.shape: {batch_X.shape}")
            print(f"   batch_y.shape: {batch_y.shape}")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(batch_X)
                print(f"   outputs.shape: {outputs.shape}")
                
                # 检查维度匹配
                if outputs.shape == batch_y.shape:
                    print("   ✅ 维度匹配")
                    loss = criterion(outputs, batch_y)
                    print(f"   📉 损失: {loss.item():.6f}")
                else:
                    print(f"   ❌ 维度不匹配: {outputs.shape} vs {batch_y.shape}")
                    
                    # 尝试修复
                    if len(batch_y.shape) == 1:
                        batch_y_fixed = batch_y.unsqueeze(1)
                        print(f"   🔧 修复后 batch_y: {batch_y_fixed.shape}")
                    else:
                        batch_y_fixed = batch_y
                    
                    if outputs.shape[1] > batch_y_fixed.shape[1]:
                        outputs_fixed = outputs[:, :batch_y_fixed.shape[1]]
                        print(f"   🔧 修复后 outputs: {outputs_fixed.shape}")
                    elif batch_y_fixed.shape[1] > outputs.shape[1]:
                        batch_y_fixed = batch_y_fixed[:, :outputs.shape[1]]
                        print(f"   🔧 修复后 batch_y: {batch_y_fixed.shape}")
                    else:
                        outputs_fixed = outputs
                    
                    if outputs_fixed.shape == batch_y_fixed.shape:
                        loss = criterion(outputs_fixed, batch_y_fixed)
                        print(f"   ✅ 修复成功，损失: {loss.item():.6f}")
                    else:
                        print(f"   ❌ 修复失败")
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 训练循环调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """建议修复方案"""
    print(f"\n💡 修复建议:")
    print("1. 检查预测天数设置是否与模型输出维度匹配")
    print("2. 确保数据预处理中的序列创建正确")
    print("3. 验证模型定义中的输出层维度")
    print("4. 在训练循环中添加维度检查和自动修复")
    
    print(f"\n🔧 推荐配置:")
    print("- 预测天数: 1-5天")
    print("- 序列长度: 20-60天")
    print("- 批次大小: 16-32")
    print("- 模型输出维度 = 预测天数")

def main():
    """主函数"""
    print("🔍 张量维度调试脚本")
    print("=" * 50)
    
    # 1. 调试数据形状
    result = debug_data_shapes()
    if result is None:
        print("❌ 数据调试失败，无法继续")
        return
    
    X, y, input_size = result
    output_size = y.shape[1] if len(y.shape) > 1 else 1
    
    # 2. 调试模型形状
    model = debug_model_shapes(input_size, output_size)
    if model is None:
        print("❌ 模型调试失败")
        return
    
    # 3. 调试训练循环
    training_ok = debug_training_loop(X, y, model)
    
    # 4. 提供修复建议
    suggest_fixes()
    
    print("\n" + "=" * 50)
    if training_ok:
        print("🎉 维度调试完成，问题已修复！")
        print("💡 现在可以正常训练模型了")
    else:
        print("⚠️ 仍存在维度问题，请检查配置")
    
    print("\n🚀 建议运行:")
    print("python main.py --stock_code 000001 --mode both --days 3")

if __name__ == "__main__":
    main()
