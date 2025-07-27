"""
数据可视化模块
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import os
from config import PATHS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StockVisualizer:
    def __init__(self):
        """初始化可视化器"""
        os.makedirs(PATHS['results_dir'], exist_ok=True)
    
    def plot_stock_data(self, df, stock_code, title=None):
        """
        绘制股票K线图
        
        Args:
            df: 股票数据
            stock_code: 股票代码
            title: 图表标题
        """
        if title is None:
            title = f"{stock_code} 股票K线图"
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('价格', '成交量', '技术指标'),
            row_width=[0.2, 0.1, 0.1]
        )
        
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="K线"
            ),
            row=1, col=1
        )
        
        # 移动平均线
        if 'ma5' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ma5'], name='MA5', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        if 'ma20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['ma20'], name='MA20', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # 成交量
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color=colors),
            row=2, col=1
        )
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue', width=1)),
                row=3, col=1
            )
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red', width=1)),
                row=3, col=1
            )
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        # 保存图片
        fig.write_html(os.path.join(PATHS['results_dir'], f'{stock_code}_kline.html'))
        fig.show()
    
    def plot_technical_indicators(self, df, stock_code):
        """
        绘制技术指标
        
        Args:
            df: 包含技术指标的数据
            stock_code: 股票代码
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{stock_code} 技术指标分析', fontsize=16)
        
        # RSI
        if 'rsi' in df.columns:
            axes[0, 0].plot(df.index, df['rsi'], label='RSI', color='purple')
            axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线')
            axes[0, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线')
            axes[0, 0].set_title('RSI指标')
            axes[0, 0].set_ylabel('RSI')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # KDJ
        if all(col in df.columns for col in ['k', 'd', 'j']):
            axes[0, 1].plot(df.index, df['k'], label='K', color='blue')
            axes[0, 1].plot(df.index, df['d'], label='D', color='red')
            axes[0, 1].plot(df.index, df['j'], label='J', color='green')
            axes[0, 1].set_title('KDJ指标')
            axes[0, 1].set_ylabel('KDJ')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 布林带
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            axes[1, 0].plot(df.index, df['close'], label='收盘价', color='black', linewidth=1)
            axes[1, 0].plot(df.index, df['bb_upper'], label='上轨', color='red', alpha=0.7)
            axes[1, 0].plot(df.index, df['bb_middle'], label='中轨', color='blue', alpha=0.7)
            axes[1, 0].plot(df.index, df['bb_lower'], label='下轨', color='green', alpha=0.7)
            axes[1, 0].fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
            axes[1, 0].set_title('布林带')
            axes[1, 0].set_ylabel('价格')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 成交量比率
        if 'volume_ratio' in df.columns:
            axes[1, 1].bar(df.index, df['volume_ratio'], alpha=0.7, color='orange')
            axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='基准线')
            axes[1, 1].set_title('成交量比率')
            axes[1, 1].set_ylabel('比率')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_technical_indicators.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_results(self, actual_prices, predicted_prices, dates, stock_code):
        """
        绘制预测结果对比
        
        Args:
            actual_prices: 实际价格
            predicted_prices: 预测价格
            dates: 日期
            stock_code: 股票代码
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(dates, actual_prices, label='实际价格', marker='o', linewidth=2)
        plt.plot(dates, predicted_prices, label='预测价格', marker='s', linewidth=2, linestyle='--')
        
        plt.title(f'{stock_code} 价格预测对比', fontsize=16)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 添加误差带
        error = np.abs(actual_prices - predicted_prices)
        plt.fill_between(dates, predicted_prices - error, predicted_prices + error, 
                        alpha=0.2, label='误差范围')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_prediction_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_future_prediction(self, historical_data, prediction_result, stock_code):
        """
        绘制未来预测
        
        Args:
            historical_data: 历史数据
            prediction_result: 预测结果字典
            stock_code: 股票代码
        """
        fig = go.Figure()
        
        # 历史数据
        recent_data = historical_data.tail(60)  # 显示最近60天
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['close'],
                mode='lines',
                name='历史价格',
                line=dict(color='blue', width=2)
            )
        )
        
        # 预测数据
        pred_dates = prediction_result['dates']
        pred_prices = prediction_result['predictions']
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='预测价格',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            )
        )
        
        # 连接线
        fig.add_trace(
            go.Scatter(
                x=[recent_data.index[-1], pred_dates[0]],
                y=[recent_data['close'].iloc[-1], pred_prices[0]],
                mode='lines',
                name='连接线',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            )
        )
        
        # 添加置信区间（简单估计）
        confidence_interval = np.std(pred_prices) * 0.5
        fig.add_trace(
            go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=list(pred_prices + confidence_interval) + list(pred_prices - confidence_interval)[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='置信区间',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title=f'{stock_code} 股价预测',
            xaxis_title='日期',
            yaxis_title='价格',
            hovermode='x unified',
            height=600
        )
        
        # 保存图片
        fig.write_html(os.path.join(PATHS['results_dir'], f'{stock_code}_future_prediction.html'))
        fig.show()
        
        # 打印预测摘要
        print(f"\n{stock_code} 预测摘要:")
        print(f"当前价格: {prediction_result['last_price']:.2f}")
        print(f"预测价格范围: {min(pred_prices):.2f} - {max(pred_prices):.2f}")
        print(f"预测涨跌: {pred_prices[-1] - prediction_result['last_price']:.2f} "
              f"({((pred_prices[-1] - prediction_result['last_price']) / prediction_result['last_price'] * 100):+.2f}%)")
    
    def plot_correlation_matrix(self, df, stock_code):
        """
        绘制特征相关性矩阵
        
        Args:
            df: 数据DataFrame
            stock_code: 股票代码
        """
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'{stock_code} 特征相关性矩阵', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_scores, stock_code):
        """
        绘制特征重要性
        
        Args:
            feature_names: 特征名称列表
            importance_scores: 重要性分数
            stock_code: 股票代码
        """
        # 排序
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_features)), sorted_scores)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('重要性分数')
        plt.title(f'{stock_code} 特征重要性')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
