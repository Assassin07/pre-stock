"""
测试技术指标库修复是否有效
"""

import warnings
warnings.filterwarnings('ignore')

def test_technical_indicators():
    """测试技术指标计算功能"""
    print("🧪 测试技术指标库修复...")
    
    # 测试导入
    talib_available = False
    ta_available = False
    
    try:
        import talib
        talib_available = True
        print("✅ TA-Lib 可用")
    except ImportError:
        print("❌ TA-Lib 不可用")
    
    try:
        import ta
        ta_available = True
        print("✅ ta 库可用")
    except ImportError:
        print("❌ ta 库不可用")
    
    if not talib_available and not ta_available:
        print("⚠️ 技术指标库都不可用，将使用简化版本")
    
    # 测试数据预处理器
    print("\n🔧 测试数据预处理器...")
    
    try:
        from data_preprocessor import StockDataPreprocessor
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # 测试预处理器
        preprocessor = StockDataPreprocessor()
        df_with_indicators = preprocessor.add_technical_indicators(test_data)
        
        print(f"✅ 技术指标计算成功，添加了 {len(df_with_indicators.columns) - len(test_data.columns)} 个指标")
        print(f"📊 指标列表: {list(df_with_indicators.columns)}")
        
        # 检查是否有NaN值
        nan_count = df_with_indicators.isnull().sum().sum()
        print(f"📈 NaN值数量: {nan_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据预处理器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_stock_data_fetch():
    """测试股票数据获取"""
    print("\n📊 测试股票数据获取...")
    
    try:
        from data_fetcher import StockDataFetcher
        
        fetcher = StockDataFetcher()
        
        # 测试获取少量数据
        df = fetcher.fetch_stock_data('000001', start_date='2023-01-01', end_date='2023-01-31')
        
        if df is not None and len(df) > 0:
            print(f"✅ 股票数据获取成功，共 {len(df)} 条记录")
            print(f"📅 数据时间范围: {df.index[0]} 到 {df.index[-1]}")
            return True
        else:
            print("❌ 股票数据获取失败")
            return False
            
    except Exception as e:
        print(f"❌ 股票数据获取测试失败: {str(e)}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🤖 测试模型创建...")
    
    try:
        from model import create_model
        
        # 测试创建LSTM模型
        model = create_model('lstm', input_size=10, output_size=3)
        print("✅ LSTM模型创建成功")
        
        # 测试模型前向传播
        import torch
        test_input = torch.randn(2, 20, 10)  # batch_size=2, seq_len=20, input_size=10
        output = model(test_input)
        print(f"✅ 模型前向传播成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("🚀 技术指标库修复测试")
    print("=" * 50)
    
    results = []
    
    # 测试技术指标
    results.append(("技术指标库", test_technical_indicators()))
    
    # 测试股票数据获取
    results.append(("股票数据获取", test_stock_data_fetch()))
    
    # 测试模型创建
    results.append(("模型创建", test_model_creation()))
    
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
    
    if passed == total:
        print("🎉 所有测试通过！修复成功")
        print("💡 现在可以正常使用股票预测系统了")
    else:
        print("⚠️ 部分测试失败，但系统仍可运行")
        print("💡 即使技术指标库不可用，系统会使用简化版本")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ 修复验证完成，系统可以正常使用")
        print("🚀 运行 'python main.py --stock_code 000001 --mode both --days 3' 开始预测")
    else:
        print("\n⚠️ 部分功能可能受限，但基本功能仍可使用")
        print("💡 建议使用 'python install_dependencies.py' 重新安装依赖")
