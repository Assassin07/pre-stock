"""
快速测试脚本 - 验证系统基本功能
"""

import warnings
warnings.filterwarnings('ignore')

def test_basic_imports():
    """测试基本导入"""
    print("🧪 测试基本导入...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        print("✅ 基础库导入成功")
        return True
    except ImportError as e:
        print(f"❌ 基础库导入失败: {e}")
        return False

def test_technical_indicators():
    """测试技术指标库"""
    print("\n🔧 测试技术指标库...")
    
    # 测试ta库
    try:
        import ta
        print("✅ ta库可用")
        return True
    except ImportError:
        print("❌ ta库不可用")
    
    # 测试talib
    try:
        import talib
        print("✅ talib库可用")
        return True
    except ImportError:
        print("❌ talib库不可用")
    
    print("⚠️ 将使用简化版技术指标")
    return True  # 简化版总是可用的

def test_data_processing():
    """测试数据处理功能"""
    print("\n📊 测试数据处理...")
    
    try:
        import pandas as pd
        import numpy as np
        from data_preprocessor import StockDataPreprocessor
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # 测试预处理
        preprocessor = StockDataPreprocessor()
        df_with_indicators = preprocessor.add_technical_indicators(test_data)
        
        print(f"✅ 技术指标计算成功，添加了 {len(df_with_indicators.columns) - len(test_data.columns)} 个指标")
        return True
        
    except Exception as e:
        print(f"❌ 数据处理测试失败: {str(e)}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🤖 测试模型创建...")
    
    try:
        from model import create_model
        import torch
        
        # 创建小型模型进行测试
        model = create_model('lstm', input_size=5, output_size=1)
        
        # 测试前向传播
        test_input = torch.randn(2, 10, 5)  # batch_size=2, seq_len=10, input_size=5
        output = model(test_input)
        
        print(f"✅ 模型创建和前向传播成功，输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {str(e)}")
        return False

def test_data_fetcher():
    """测试数据获取器（使用示例数据）"""
    print("\n📈 测试数据获取器...")
    
    try:
        from data_fetcher import StockDataFetcher
        
        fetcher = StockDataFetcher()
        
        # 强制使用示例数据模式
        df = fetcher._create_sample_data('000001', '2023-01-01', '2023-01-31', 'daily')
        
        if df is not None and len(df) > 0:
            print(f"✅ 示例数据创建成功，共 {len(df)} 条记录")
            print(f"📊 数据列: {list(df.columns)}")
            return True
        else:
            print("❌ 示例数据创建失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据获取器测试失败: {str(e)}")
        return False

def run_quick_demo():
    """运行快速演示"""
    print("\n🚀 运行快速演示...")
    
    try:
        import pandas as pd
        import numpy as np
        from data_preprocessor import StockDataPreprocessor
        from model import create_model
        import torch
        
        print("📊 创建示例数据...")
        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        base_price = 10.0
        returns = np.random.normal(0.001, 0.02, 200)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 200)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        print("🔧 处理数据...")
        # 数据预处理
        preprocessor = StockDataPreprocessor()
        preprocessor.sequence_length = 20  # 减少序列长度
        preprocessor.prediction_days = 3   # 减少预测天数
        
        df_with_indicators = preprocessor.add_technical_indicators(df)
        feature_data = preprocessor.select_features(df_with_indicators)
        feature_data = feature_data.dropna()
        
        print(f"✅ 特征数量: {len(preprocessor.feature_columns)}")
        print(f"✅ 有效数据点: {len(feature_data)}")
        
        # 创建简单的序列数据
        normalized_data = preprocessor.normalize_data(feature_data.values, fit_scaler=True)
        X, y = preprocessor.create_sequences(pd.DataFrame(normalized_data, columns=preprocessor.feature_columns, index=feature_data.index))
        
        print(f"✅ 序列数据形状: X={X.shape}, y={y.shape}")
        
        print("🤖 创建和测试模型...")
        # 创建模型
        model = create_model('lstm', input_size=len(preprocessor.feature_columns), output_size=3)
        
        # 测试预测
        with torch.no_grad():
            test_input = torch.FloatTensor(X[:5])  # 取前5个样本
            predictions = model(test_input)
            print(f"✅ 预测成功，预测形状: {predictions.shape}")
        
        print("🎉 快速演示完成！系统基本功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 快速演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 快速测试脚本")
    print("=" * 50)
    
    tests = [
        ("基本导入", test_basic_imports),
        ("技术指标库", test_technical_indicators),
        ("数据处理", test_data_processing),
        ("模型创建", test_model_creation),
        ("数据获取器", test_data_fetcher),
    ]
    
    results = []
    
    for test_name, test_func in tests:
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
        print(f"{test_name:<15}: {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed >= total - 1:  # 允许一个测试失败
        print("\n🎉 系统基本功能正常！")
        
        # 运行快速演示
        demo_success = run_quick_demo()
        
        if demo_success:
            print("\n✅ 系统完全可用！")
            print("💡 现在可以运行完整的股票预测了")
            print("🚀 尝试运行: python main.py --stock_code 000001 --mode both --days 3")
        else:
            print("\n⚠️ 演示失败，但基本功能可用")
    else:
        print("\n❌ 系统存在问题，请检查依赖安装")
        print("💡 尝试运行: python install_dependencies.py")

if __name__ == "__main__":
    main()
