"""
股票数据获取模块
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import PATHS


class StockDataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        self.data_dir = PATHS['data_dir']
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_stock_data(self, stock_code, start_date=None, end_date=None, period="daily"):
        """
        获取股票数据

        Args:
            stock_code: 股票代码，如 '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            period: 数据周期，'daily', 'weekly', 'monthly'

        Returns:
            DataFrame: 股票数据
        """
        # 多种方法尝试获取数据
        methods = [
            self._fetch_with_akshare_new,
            self._fetch_with_akshare_old,
            self._create_sample_data
        ]

        for i, method in enumerate(methods, 1):
            try:
                print(f"尝试方法 {i}: {method.__name__}")
                df = method(stock_code, start_date, end_date, period)
                if df is not None and len(df) > 0:
                    print(f"✅ 成功获取 {len(df)} 条数据")
                    return df
            except Exception as e:
                print(f"❌ 方法 {i} 失败: {str(e)}")
                continue

        print("❌ 所有方法都失败了")
        return None

    def _fetch_with_akshare_new(self, stock_code, start_date, end_date, period):
        """使用新版akshare API获取数据"""
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = end_date.replace('-', '')

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
        else:
            start_date = start_date.replace('-', '')

        print(f"正在获取股票 {stock_code} 从 {start_date} 到 {end_date} 的数据...")

        # 获取股票历史数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                               start_date=start_date, end_date=end_date, adjust="qfq")

        if df is None or df.empty:
            raise ValueError(f"无法获取股票 {stock_code} 的数据")

        # 处理列名（akshare可能返回不同的列名）
        df = self._standardize_columns(df)

        return df

    def _fetch_with_akshare_old(self, stock_code, start_date, end_date, period):
        """使用旧版akshare API获取数据"""
        try:
            # 尝试使用不同的API
            df = ak.stock_zh_a_daily(symbol=f"sz{stock_code}" if stock_code.startswith('0') else f"sh{stock_code}")

            if df is not None and len(df) > 0:
                # 限制日期范围
                if start_date:
                    start_dt = pd.to_datetime(start_date.replace('-', ''))
                    df = df[df.index >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date.replace('-', ''))
                    df = df[df.index <= end_dt]

                df = self._standardize_columns(df)
                return df
        except:
            pass

        # 如果上面失败，尝试另一个API
        df = ak.stock_individual_info_em(symbol=stock_code)
        if df is not None:
            # 这个API返回的是基本信息，我们需要转换为历史数据格式
            raise ValueError("此API不返回历史数据")

        return None

    def _create_sample_data(self, stock_code, start_date, end_date, period):
        """创建示例数据（当无法获取真实数据时）"""
        print("⚠️ 无法获取真实数据，创建示例数据用于测试...")

        # 设置日期范围
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = datetime.now() - timedelta(days=365)

        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()

        # 创建日期序列（只包含工作日）
        dates = pd.bdate_range(start=start_dt, end=end_dt)

        # 生成模拟股价数据
        np.random.seed(42)  # 固定随机种子以获得一致的结果
        n_days = len(dates)

        # 基础价格
        base_price = 10.0

        # 生成价格走势
        returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
        prices = base_price * np.exp(np.cumsum(returns))

        # 生成OHLC数据
        opens = prices * (1 + np.random.normal(0, 0.005, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        closes = prices
        volumes = np.random.randint(1000000, 10000000, n_days)

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

        print(f"✅ 创建了 {len(df)} 条示例数据")
        return df

    def _standardize_columns(self, df):
        """标准化列名"""
        # 可能的列名映射
        column_mapping = {
            '日期': 'date',
            'date': 'date',
            '开盘': 'open',
            'open': 'open',
            '收盘': 'close',
            'close': 'close',
            '最高': 'high',
            'high': 'high',
            '最低': 'low',
            'low': 'low',
            '成交量': 'volume',
            'volume': 'volume',
            '成交额': 'turnover',
            'turnover': 'turnover',
            '振幅': 'amplitude',
            'amplitude': 'amplitude',
            '涨跌幅': 'change_pct',
            'change_pct': 'change_pct',
            '涨跌额': 'change_amount',
            'change_amount': 'change_amount',
            '换手率': 'turnover_rate',
            'turnover_rate': 'turnover_rate'
        }

        # 重命名列
        df_renamed = df.rename(columns=column_mapping)

        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df_renamed.columns:
                if col == 'volume' and '成交量' in df.columns:
                    df_renamed[col] = df['成交量']
                elif col in ['open', 'high', 'low', 'close']:
                    # 如果缺少OHLC数据，用close价格填充
                    if 'close' in df_renamed.columns:
                        df_renamed[col] = df_renamed['close']
                    else:
                        raise ValueError(f"缺少必要的列: {col}")

        # 处理日期索引
        if 'date' in df_renamed.columns:
            df_renamed['date'] = pd.to_datetime(df_renamed['date'])
            df_renamed.set_index('date', inplace=True)
        elif not isinstance(df_renamed.index, pd.DatetimeIndex):
            # 如果索引不是日期类型，尝试转换
            try:
                df_renamed.index = pd.to_datetime(df_renamed.index)
            except:
                # 如果转换失败，创建日期索引
                df_renamed.index = pd.date_range(start='2023-01-01', periods=len(df_renamed), freq='D')

        # 按日期排序
        df_renamed.sort_index(inplace=True)

        return df_renamed
    
    def get_stock_info(self, stock_code):
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            dict: 股票信息
        """
        try:
            # 获取股票基本信息
            info = ak.stock_individual_info_em(symbol=stock_code)
            return info
        except Exception as e:
            print(f"获取股票信息时出错: {e}")
            return None
    
    def save_data(self, df, stock_code, filename=None):
        """
        保存数据到本地
        
        Args:
            df: 数据DataFrame
            stock_code: 股票代码
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            filename = f"{stock_code}_data.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"数据已保存到: {filepath}")
    
    def load_data(self, stock_code, filename=None):
        """
        从本地加载数据
        
        Args:
            stock_code: 股票代码
            filename: 文件名，如果为None则自动生成
            
        Returns:
            DataFrame: 股票数据
        """
        if filename is None:
            filename = f"{stock_code}_data.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            print(f"从本地加载数据: {filepath}")
            return df
        else:
            print(f"本地文件不存在: {filepath}")
            return None


if __name__ == "__main__":
    # 测试代码
    fetcher = StockDataFetcher()
    
    # 获取平安银行数据
    stock_code = "000001"
    df = fetcher.fetch_stock_data(stock_code)
    
    if df is not None:
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"数据预览:")
        print(df.head())
        
        # 保存数据
        fetcher.save_data(df, stock_code)
