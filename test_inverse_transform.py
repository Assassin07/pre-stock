"""
测试反标准化功能
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def test_inverse_transform():
    """测试反标准化功能"""
    print("🧪 测试反标准化功能...")
    
    try:
        from data_preprocessor import StockDataPreprocessor
        
        # 创建测试数据
        print("📊 创建测试数据...")
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # 生成模拟股价数据
        base_price = 10.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        print(f"✅ 测试数据: {df.shape}")
        
        # 数据预处理
        print("🔧 数据预处理...")
        preprocessor = StockDataPreprocessor()
        
        # 添加技术指标
        df_with_indicators = preprocessor.add_technical_indicators(df)
        feature_data = preprocessor.select_features(df_with_indicators)
        feature_data = feature_data.dropna()
        
        print(f"📈 特征数据: {feature_data.shape}")
        print(f"📋 特征列: {preprocessor.feature_columns}")
        
        # 标准化
        original_data = feature_data.values
        normalized_data = preprocessor.normalize_data(original_data, fit_scaler=True)
        
        print(f"📏 标准化数据: {normalized_data.shape}")
        
        # 测试反标准化
        print("\n🔄 测试反标准化...")
        
        # 测试1: 反标准化单列数据
        print("测试1: 单列数据反标准化")
        close_col_idx = preprocessor.feature_columns.index('close') if 'close' in preprocessor.feature_columns else 0
        close_data_normalized = normalized_data[:10, close_col_idx]  # 取前10个样本的close列
        
        print(f"   输入形状: {close_data_normalized.shape}")
        
        # 使用列名
        result1 = preprocessor.inverse_transform(close_data_normalized, target_column='close')
        print(f"   结果1形状: {result1.shape}")
        
        # 使用索引
        result2 = preprocessor.inverse_transform(close_data_normalized, target_column=close_col_idx)
        print(f"   结果2形状: {result2.shape}")
        
        # 验证结果
        original_close = original_data[:10, close_col_idx]
        error1 = np.mean(np.abs(result1 - original_close))
        error2 = np.mean(np.abs(result2 - original_close))
        
        print(f"   误差1: {error1:.6f}")
        print(f"   误差2: {error2:.6f}")
        
        if error1 < 1e-10 and error2 < 1e-10:
            print("   ✅ 单列反标准化测试通过")
        else:
            print("   ❌ 单列反标准化测试失败")
        
        # 测试2: 反标准化多维数据
        print("\n测试2: 多维数据反标准化")
        multi_data = np.random.randn(5, 3)  # 5个样本，3个预测天数
        
        print(f"   输入形状: {multi_data.shape}")
        
        result3 = preprocessor.inverse_transform(multi_data, target_column='close')
        print(f"   结果形状: {result3.shape}")
        
        if result3.shape == multi_data.shape:
            print("   ✅ 多维反标准化形状正确")
        else:
            print("   ❌ 多维反标准化形状错误")
        
        # 测试3: 错误情况处理
        print("\n测试3: 错误情况处理")
        
        # 测试不存在的列名
        result4 = preprocessor.inverse_transform(close_data_normalized, target_column='nonexistent')
        print(f"   不存在列名结果形状: {result4.shape}")
        
        # 测试超出范围的索引
        result5 = preprocessor.inverse_transform(close_data_normalized, target_column=999)
        print(f"   超出范围索引结果形状: {result5.shape}")
        
        print("✅ 反标准化功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 反标准化测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_evaluation():
    """测试预测器评估功能"""
    print("\n🔮 测试预测器评估...")
    
    try:
        from data_preprocessor import StockDataPreprocessor
        from predictor import StockPredictor
        from model import create_model
        import torch
        
        # 创建测试数据
        print("📊 准备测试数据...")
        preprocessor = StockDataPreprocessor()
        
        # 模拟已经处理好的数据
        preprocessor.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 创建模拟的标准化数据
        X_test = np.random.randn(20, 30, 5)  # 20个样本，30个时间步，5个特征
        y_test = np.random.randn(20, 3)      # 20个样本，3天预测
        
        print(f"   X_test形状: {X_test.shape}")
        print(f"   y_test形状: {y_test.shape}")
        
        # 创建预测器
        predictor = StockPredictor('lstm', input_size=5, output_size=3)
        predictor.preprocessor = preprocessor
        
        # 模拟缩放器
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        dummy_data = np.random.randn(100, 5)
        scaler.fit(dummy_data)
        preprocessor.scaler = scaler
        
        # 测试评估
        print("🔍 测试评估功能...")
        metrics = predictor.evaluate((X_test, y_test), 'TEST')
        
        print("✅ 评估功能测试完成")
        print(f"📊 评估指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测器评估测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """端到端测试"""
    print("\n🚀 端到端测试...")
    
    try:
        from data_fetcher import StockDataFetcher
        from data_preprocessor import StockDataPreprocessor
        from trainer import StockTrainer
        from predictor import StockPredictor
        
        # 获取测试数据
        print("📊 获取测试数据...")
        fetcher = StockDataFetcher()
        df = fetcher._create_sample_data('TEST', '2023-01-01', '2023-03-31', 'daily')
        
        if df is None or len(df) < 50:
            print("❌ 测试数据不足")
            return False
        
        # 数据预处理
        print("🔧 数据预处理...")
        preprocessor = StockDataPreprocessor()
        preprocessor.sequence_length = 20
        preprocessor.prediction_days = 3
        
        train_data, val_data, test_data = preprocessor.prepare_data(df)
        input_size = len(preprocessor.feature_columns)
        
        print(f"✅ 数据准备完成，特征数: {input_size}")
        
        # 快速训练
        print("🤖 快速训练...")
        trainer = StockTrainer('lstm', input_size, 3)
        
        # 修改配置以加快测试
        from config import TRAINING_CONFIG
        original_epochs = TRAINING_CONFIG['num_epochs']
        TRAINING_CONFIG['num_epochs'] = 2
        
        try:
            train_losses, val_losses = trainer.train(train_data, val_data, 'TEST')
            print("✅ 训练完成")
        finally:
            TRAINING_CONFIG['num_epochs'] = original_epochs
        
        # 测试预测和评估
        print("🔮 测试预测和评估...")
        predictor = StockPredictor('lstm', input_size, 3)
        predictor.load_model('TEST')
        predictor.preprocessor = preprocessor
        
        # 评估
        metrics = predictor.evaluate(test_data, 'TEST')
        print("✅ 评估完成")
        
        # 预测未来
        future_result = predictor.predict_next_days(df, 'TEST', 3)
        print("✅ 未来预测完成")
        
        print("🎉 端到端测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 反标准化功能测试")
    print("=" * 50)
    
    tests = [
        ("反标准化功能", test_inverse_transform),
        ("预测器评估", test_predictor_evaluation),
        ("端到端测试", test_end_to_end),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {str(e)}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！反标准化问题已修复")
        print("💡 现在可以正常使用股票预测系统了")
    else:
        print("\n⚠️ 部分测试失败，但基本功能可用")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 建议运行完整测试:")
        print("python main.py --stock_code 000001 --mode both --days 3")
    else:
        print("\n💡 如果问题持续存在，请检查:")
        print("1. 特征列配置是否正确")
        print("2. 缩放器是否正确保存和加载")
        print("3. 数据维度是否匹配")
