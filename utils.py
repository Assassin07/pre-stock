"""
工具函数模块
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import PATHS


def setup_logging(log_level=logging.INFO):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_prediction.log'),
            logging.StreamHandler()
        ]
    )


def create_directories():
    """创建必要的目录"""
    print("📁 创建项目目录结构...")

    for name, path in PATHS.items():
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.exists(path):
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: {path} - 创建失败")
        except Exception as e:
            print(f"❌ 创建目录失败 {path}: {str(e)}")
            # 尝试使用绝对路径
            try:
                abs_path = os.path.abspath(path)
                os.makedirs(abs_path, exist_ok=True)
                print(f"🔧 使用绝对路径创建: {abs_path}")
            except Exception as e2:
                print(f"❌ 绝对路径也失败: {str(e2)}")

    print("📋 目录检查完成")


def save_json(data, filename, directory=None):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        filename: 文件名
        directory: 目录路径
    """
    if directory is None:
        directory = PATHS['results_dir']
    
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON文件已保存: {filepath}")


def load_json(filename, directory=None):
    """
    加载JSON文件
    
    Args:
        filename: 文件名
        directory: 目录路径
        
    Returns:
        dict: 加载的数据
    """
    if directory is None:
        directory = PATHS['results_dir']
    
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"JSON文件已加载: {filepath}")
        return data
    else:
        print(f"JSON文件不存在: {filepath}")
        return None


def save_pickle(data, filename, directory=None):
    """
    保存Pickle文件
    
    Args:
        data: 要保存的数据
        filename: 文件名
        directory: 目录路径
    """
    if directory is None:
        directory = PATHS['model_dir']
    
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Pickle文件已保存: {filepath}")


def load_pickle(filename, directory=None):
    """
    加载Pickle文件
    
    Args:
        filename: 文件名
        directory: 目录路径
        
    Returns:
        object: 加载的数据
    """
    if directory is None:
        directory = PATHS['model_dir']
    
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Pickle文件已加载: {filepath}")
        return data
    else:
        print(f"Pickle文件不存在: {filepath}")
        return None


def calculate_returns(prices):
    """
    计算收益率
    
    Args:
        prices: 价格序列
        
    Returns:
        numpy.ndarray: 收益率序列
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    returns = np.diff(prices) / prices[:-1]
    return returns


def calculate_volatility(returns, window=20):
    """
    计算波动率
    
    Args:
        returns: 收益率序列
        window: 滚动窗口大小
        
    Returns:
        numpy.ndarray: 波动率序列
    """
    if isinstance(returns, pd.Series):
        volatility = returns.rolling(window=window).std()
    else:
        volatility = pd.Series(returns).rolling(window=window).std().values
    
    return volatility


def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        float: 夏普比率
    """
    excess_returns = returns - risk_free_rate / 252  # 假设252个交易日
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio


def calculate_max_drawdown(prices):
    """
    计算最大回撤
    
    Args:
        prices: 价格序列
        
    Returns:
        float: 最大回撤
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return max_drawdown


def validate_stock_code(stock_code):
    """
    验证股票代码格式
    
    Args:
        stock_code: 股票代码
        
    Returns:
        bool: 是否有效
    """
    if not isinstance(stock_code, str):
        return False
    
    # A股股票代码格式验证
    if len(stock_code) == 6 and stock_code.isdigit():
        return True
    
    return False


def get_trading_dates(start_date, end_date):
    """
    获取交易日期列表（简化版本，实际应该考虑节假日）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        list: 交易日期列表
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    trading_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # 排除周末
        if current_date.weekday() < 5:
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_dates


def normalize_features(data, method='minmax'):
    """
    特征标准化
    
    Args:
        data: 输入数据
        method: 标准化方法 ('minmax', 'zscore')
        
    Returns:
        tuple: (标准化后的数据, 标准化参数)
    """
    if method == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)
        params = {'min_vals': min_vals, 'max_vals': max_vals}
    elif method == 'zscore':
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        normalized_data = (data - mean_vals) / (std_vals + 1e-8)
        params = {'mean_vals': mean_vals, 'std_vals': std_vals}
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized_data, params


def denormalize_features(data, params, method='minmax'):
    """
    特征反标准化
    
    Args:
        data: 标准化的数据
        params: 标准化参数
        method: 标准化方法
        
    Returns:
        numpy.ndarray: 反标准化后的数据
    """
    if method == 'minmax':
        min_vals = params['min_vals']
        max_vals = params['max_vals']
        denormalized_data = data * (max_vals - min_vals) + min_vals
    elif method == 'zscore':
        mean_vals = params['mean_vals']
        std_vals = params['std_vals']
        denormalized_data = data * std_vals + mean_vals
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    return denormalized_data


def calculate_technical_signals(df):
    """
    计算技术分析信号
    
    Args:
        df: 包含技术指标的数据
        
    Returns:
        dict: 技术信号
    """
    signals = {}
    
    # RSI信号
    if 'rsi' in df.columns:
        latest_rsi = df['rsi'].iloc[-1]
        if latest_rsi > 70:
            signals['rsi'] = '超买'
        elif latest_rsi < 30:
            signals['rsi'] = '超卖'
        else:
            signals['rsi'] = '中性'
    
    # MACD信号
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        latest_macd = df['macd'].iloc[-1]
        latest_signal = df['macd_signal'].iloc[-1]
        if latest_macd > latest_signal:
            signals['macd'] = '看涨'
        else:
            signals['macd'] = '看跌'
    
    # 移动平均线信号
    if 'ma5' in df.columns and 'ma20' in df.columns:
        latest_ma5 = df['ma5'].iloc[-1]
        latest_ma20 = df['ma20'].iloc[-1]
        if latest_ma5 > latest_ma20:
            signals['ma'] = '看涨'
        else:
            signals['ma'] = '看跌'
    
    return signals


def print_model_summary(model):
    """
    打印模型摘要信息
    
    Args:
        model: PyTorch模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型摘要:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型结构:")
    print(model)


def format_number(num, precision=2):
    """
    格式化数字显示
    
    Args:
        num: 数字
        precision: 精度
        
    Returns:
        str: 格式化后的字符串
    """
    if abs(num) >= 1e8:
        return f"{num/1e8:.{precision}f}亿"
    elif abs(num) >= 1e4:
        return f"{num/1e4:.{precision}f}万"
    else:
        return f"{num:.{precision}f}"


if __name__ == "__main__":
    # 测试工具函数
    create_directories()
    print("工具函数测试完成")
