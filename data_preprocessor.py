"""
数据预处理模块
"""

import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from config import DATA_CONFIG, PATHS


class StockDataPreprocessor:
    def __init__(self):
        """初始化数据预处理器"""
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.sequence_length = DATA_CONFIG['sequence_length']
        self.prediction_days = DATA_CONFIG['prediction_days']
        
    def add_technical_indicators(self, df):
        """
        添加技术指标
        
        Args:
            df: 原始股票数据
            
        Returns:
            DataFrame: 添加技术指标后的数据
        """
        df = df.copy()
        
        # 基础价格数据
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        # 移动平均线
        df['ma5'] = talib.SMA(close, timeperiod=5)
        df['ma10'] = talib.SMA(close, timeperiod=10)
        df['ma20'] = talib.SMA(close, timeperiod=20)
        df['ma60'] = talib.SMA(close, timeperiod=60)
        
        # 指数移动平均线
        df['ema12'] = talib.EMA(close, timeperiod=12)
        df['ema26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, timeperiod=20)
        
        # KDJ指标
        df['k'], df['d'] = talib.STOCH(high, low, close)
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 威廉指标
        df['wr'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # 成交量指标
        df['volume_ma5'] = talib.SMA(volume.astype(float), timeperiod=5)
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def select_features(self, df):
        """
        选择特征列
        
        Args:
            df: 包含技术指标的数据
            
        Returns:
            DataFrame: 选择的特征数据
        """
        # 选择用于训练的特征
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'ma60',
            'ema12', 'ema26',
            'macd', 'macd_signal', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'k', 'd', 'j', 'wr',
            'volume_ratio', 'price_change', 'high_low_ratio', 'open_close_ratio'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_columns
        
        return df[available_columns]
    
    def create_sequences(self, data, target_column='close'):
        """
        创建时间序列数据
        
        Args:
            data: 特征数据
            target_column: 目标列名
            
        Returns:
            tuple: (X, y) 序列数据和目标数据
        """
        X, y = [], []
        
        # 获取目标列的索引
        if target_column in data.columns:
            target_idx = data.columns.get_loc(target_column)
        else:
            target_idx = 3  # 默认使用close列
        
        for i in range(self.sequence_length, len(data) - self.prediction_days + 1):
            # 输入序列
            X.append(data.iloc[i-self.sequence_length:i].values)
            
            # 目标值（未来几天的收盘价）
            future_prices = data.iloc[i:i+self.prediction_days, target_idx].values
            y.append(future_prices)
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data, fit_scaler=True):
        """
        数据标准化
        
        Args:
            data: 输入数据
            fit_scaler: 是否拟合缩放器
            
        Returns:
            array: 标准化后的数据
        """
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        
        return scaled_data
    
    def inverse_transform(self, data, target_column_idx=3):
        """
        反标准化
        
        Args:
            data: 标准化的数据
            target_column_idx: 目标列索引
            
        Returns:
            array: 反标准化后的数据
        """
        # 创建与原始数据相同形状的数组
        dummy_data = np.zeros((data.shape[0], len(self.feature_columns)))
        dummy_data[:, target_column_idx] = data.flatten()
        
        # 反标准化
        inverse_data = self.scaler.inverse_transform(dummy_data)
        
        return inverse_data[:, target_column_idx].reshape(data.shape)
    
    def prepare_data(self, df, target_column='close'):
        """
        完整的数据预处理流程
        
        Args:
            df: 原始股票数据
            target_column: 目标列名
            
        Returns:
            tuple: 训练、验证、测试数据
        """
        print("开始数据预处理...")
        
        # 添加技术指标
        df_with_indicators = self.add_technical_indicators(df)
        
        # 选择特征
        feature_data = self.select_features(df_with_indicators)
        
        # 删除包含NaN的行
        feature_data = feature_data.dropna()
        
        print(f"特征数量: {len(self.feature_columns)}")
        print(f"有效数据点: {len(feature_data)}")
        
        # 数据标准化
        normalized_data = self.normalize_data(feature_data.values, fit_scaler=True)
        normalized_df = pd.DataFrame(normalized_data, columns=self.feature_columns, index=feature_data.index)
        
        # 创建序列数据
        X, y = self.create_sequences(normalized_df, target_column)
        
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 分割数据集
        train_size = int(len(X) * DATA_CONFIG['train_ratio'])
        val_size = int(len(X) * DATA_CONFIG['val_ratio'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_scaler(self, filename='scaler.pkl'):
        """保存缩放器"""
        os.makedirs(PATHS['model_dir'], exist_ok=True)
        filepath = os.path.join(PATHS['model_dir'], filename)
        joblib.dump(self.scaler, filepath)
        print(f"缩放器已保存到: {filepath}")
    
    def load_scaler(self, filename='scaler.pkl'):
        """加载缩放器"""
        filepath = os.path.join(PATHS['model_dir'], filename)
        if os.path.exists(filepath):
            self.scaler = joblib.load(filepath)
            print(f"缩放器已加载: {filepath}")
            return True
        return False
