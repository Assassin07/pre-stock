"""
网络问题修复脚本
解决akshare数据获取问题
"""

import warnings
warnings.filterwarnings('ignore')

def test_network_connection():
    """测试网络连接"""
    print("🌐 测试网络连接...")
    
    import urllib.request
    import socket
    
    test_urls = [
        "https://www.baidu.com",
        "https://www.sina.com.cn", 
        "http://push2.eastmoney.com"
    ]
    
    for url in test_urls:
        try:
            response = urllib.request.urlopen(url, timeout=5)
            if response.getcode() == 200:
                print(f"✅ {url} 连接成功")
                return True
        except Exception as e:
            print(f"❌ {url} 连接失败: {str(e)}")
    
    print("❌ 网络连接测试失败")
    return False

def test_akshare_installation():
    """测试akshare安装"""
    print("\n📦 测试akshare安装...")
    
    try:
        import akshare as ak
        print(f"✅ akshare版本: {ak.__version__}")
        return True
    except ImportError:
        print("❌ akshare未安装")
        return False
    except Exception as e:
        print(f"❌ akshare导入失败: {str(e)}")
        return False

def test_akshare_apis():
    """测试不同的akshare API"""
    print("\n🔍 测试akshare API...")
    
    try:
        import akshare as ak
        
        # 测试API 1: 股票基本信息
        try:
            print("测试API 1: 股票基本信息...")
            info = ak.stock_individual_info_em(symbol="000001")
            if info is not None:
                print("✅ 股票基本信息API可用")
                return True
        except Exception as e:
            print(f"❌ 股票基本信息API失败: {str(e)}")
        
        # 测试API 2: 股票历史数据（新版）
        try:
            print("测试API 2: 股票历史数据（新版）...")
            df = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                                   start_date="20231201", end_date="20231210", adjust="qfq")
            if df is not None and len(df) > 0:
                print("✅ 股票历史数据API（新版）可用")
                return True
        except Exception as e:
            print(f"❌ 股票历史数据API（新版）失败: {str(e)}")
        
        # 测试API 3: 实时行情
        try:
            print("测试API 3: 实时行情...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                print("✅ 实时行情API可用")
                return True
        except Exception as e:
            print(f"❌ 实时行情API失败: {str(e)}")
        
        print("❌ 所有akshare API都不可用")
        return False
        
    except Exception as e:
        print(f"❌ akshare测试失败: {str(e)}")
        return False

def create_offline_data():
    """创建离线测试数据"""
    print("\n💾 创建离线测试数据...")
    
    try:
        import pandas as pd
        import numpy as np
        import os
        
        # 创建data目录
        os.makedirs('data', exist_ok=True)
        
        # 生成多只股票的示例数据
        stock_codes = ['000001', '000002', '600036', '600519']
        
        for stock_code in stock_codes:
            # 创建一年的交易数据
            dates = pd.bdate_range(start='2023-01-01', end='2023-12-31')
            n_days = len(dates)
            
            # 设置随机种子以获得一致的数据
            np.random.seed(int(stock_code))
            
            # 基础价格
            base_price = np.random.uniform(8, 50)
            
            # 生成价格走势
            returns = np.random.normal(0.001, 0.02, n_days)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # 生成OHLC数据
            opens = prices * (1 + np.random.normal(0, 0.005, n_days))
            highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
            lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
            closes = prices
            volumes = np.random.randint(1000000, 50000000, n_days)
            
            # 创建DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'turnover': volumes * closes,
                'amplitude': (highs - lows) / closes * 100,
                'change_pct': np.concatenate([[0], np.diff(closes) / closes[:-1] * 100]),
                'change_amount': np.concatenate([[0], np.diff(closes)]),
                'turnover_rate': np.random.uniform(0.5, 5.0, n_days)
            }, index=dates)
            
            # 保存数据
            filename = f'data/{stock_code}_offline_data.csv'
            df.to_csv(filename)
            print(f"✅ 创建 {stock_code} 离线数据: {filename}")
        
        print(f"✅ 成功创建 {len(stock_codes)} 只股票的离线数据")
        return True
        
    except Exception as e:
        print(f"❌ 创建离线数据失败: {str(e)}")
        return False

def update_data_fetcher_for_offline():
    """更新数据获取器以支持离线模式"""
    print("\n🔧 配置离线模式...")
    
    try:
        # 创建离线模式配置文件
        offline_config = """
# 离线模式配置
OFFLINE_MODE = True
OFFLINE_DATA_DIR = 'data'

# 可用的离线股票代码
OFFLINE_STOCKS = ['000001', '000002', '600036', '600519']
"""
        
        with open('offline_config.py', 'w', encoding='utf-8') as f:
            f.write(offline_config)
        
        print("✅ 离线模式配置完成")
        print("💡 现在可以使用离线数据进行测试")
        return True
        
    except Exception as e:
        print(f"❌ 配置离线模式失败: {str(e)}")
        return False

def run_offline_test():
    """运行离线测试"""
    print("\n🧪 运行离线测试...")
    
    try:
        from data_fetcher import StockDataFetcher
        import pandas as pd
        import os
        
        # 检查离线数据是否存在
        offline_file = 'data/000001_offline_data.csv'
        if not os.path.exists(offline_file):
            print("❌ 离线数据不存在，请先创建")
            return False
        
        # 加载离线数据
        df = pd.read_csv(offline_file, index_col=0, parse_dates=True)
        
        if df is not None and len(df) > 0:
            print(f"✅ 离线数据加载成功，共 {len(df)} 条记录")
            print(f"📅 数据时间范围: {df.index[0].date()} 到 {df.index[-1].date()}")
            
            # 测试数据预处理
            from data_preprocessor import StockDataPreprocessor
            
            preprocessor = StockDataPreprocessor()
            df_with_indicators = preprocessor.add_technical_indicators(df)
            
            print(f"✅ 技术指标计算成功，共 {len(df_with_indicators.columns)} 列")
            
            return True
        else:
            print("❌ 离线数据加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 离线测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 网络问题修复脚本")
    print("=" * 50)
    
    # 测试网络连接
    network_ok = test_network_connection()
    
    # 测试akshare
    akshare_ok = test_akshare_installation()
    
    if akshare_ok:
        api_ok = test_akshare_apis()
    else:
        api_ok = False
    
    print("\n" + "=" * 50)
    print("📊 诊断结果")
    print("=" * 50)
    print(f"网络连接: {'✅ 正常' if network_ok else '❌ 异常'}")
    print(f"akshare安装: {'✅ 正常' if akshare_ok else '❌ 异常'}")
    print(f"akshare API: {'✅ 正常' if api_ok else '❌ 异常'}")
    
    if network_ok and akshare_ok and api_ok:
        print("\n🎉 网络和API都正常，可以正常使用在线数据")
    else:
        print("\n⚠️ 检测到网络或API问题，建议使用离线模式")
        
        # 创建离线数据
        offline_created = create_offline_data()
        
        if offline_created:
            # 配置离线模式
            offline_configured = update_data_fetcher_for_offline()
            
            if offline_configured:
                # 测试离线模式
                offline_test_ok = run_offline_test()
                
                if offline_test_ok:
                    print("\n🎉 离线模式配置成功！")
                    print("💡 现在可以使用离线数据进行股票预测")
                    print("🚀 运行: python quick_test.py 进行测试")
                else:
                    print("\n❌ 离线模式测试失败")
            else:
                print("\n❌ 离线模式配置失败")
        else:
            print("\n❌ 离线数据创建失败")
    
    print("\n💡 建议:")
    if not network_ok:
        print("- 检查网络连接")
        print("- 尝试使用VPN或代理")
    if not akshare_ok:
        print("- 重新安装akshare: pip install akshare")
    if not api_ok:
        print("- akshare API可能暂时不可用")
        print("- 使用离线模式进行测试")

if __name__ == "__main__":
    main()
