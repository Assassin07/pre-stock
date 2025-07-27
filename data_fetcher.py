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
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            else:
                end_date = end_date.replace('-', '')
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
            else:
                start_date = start_date.replace('-', '')
            
            print(f"正在获取股票 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
            
            # 获取股票历史数据
            if period == "daily":
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", 
                                       start_date=start_date, end_date=end_date, adjust="qfq")
            else:
                df = ak.stock_zh_a_hist(symbol=stock_code, period=period, 
                                       start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df is None or df.empty:
                raise ValueError(f"无法获取股票 {stock_code} 的数据")
            
            # 重命名列
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
            
            # 设置日期为索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 按日期排序
            df.sort_index(inplace=True)
            
            print(f"成功获取 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"获取股票数据时出错: {e}")
            return None
    
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
