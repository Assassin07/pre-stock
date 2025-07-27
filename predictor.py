"""
股票预测模块
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from config import DATA_CONFIG, PATHS
from model import create_model
from data_preprocessor import StockDataPreprocessor


class StockPredictor:
    def __init__(self, model_type='lstm', input_size=20, output_size=5):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型
            input_size: 输入特征数量
            output_size: 输出大小
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        
        # 创建模型
        self.model = create_model(model_type, input_size, output_size)
        self.model.to(self.device)
        
        # 数据预处理器
        self.preprocessor = StockDataPreprocessor()
        
        # 预测结果
        self.predictions = None
        self.actual_values = None
        
    def load_model(self, stock_code, is_best=True):
        """
        加载训练好的模型
        
        Args:
            stock_code: 股票代码
            is_best: 是否加载最佳模型
            
        Returns:
            bool: 是否成功加载
        """
        if is_best:
            filename = f"{stock_code}_best_model.pth"
        else:
            # 查找最新的模型文件
            model_files = [f for f in os.listdir(PATHS['model_dir']) 
                          if f.startswith(f"{stock_code}_model_epoch_")]
            if not model_files:
                return False
            filename = sorted(model_files)[-1]
        
        filepath = os.path.join(PATHS['model_dir'], filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"模型已加载: {filepath}")
            return True
        else:
            print(f"模型文件不存在: {filepath}")
            return False
    
    def predict(self, X):
        """
        进行预测
        
        Args:
            X: 输入数据 (batch_size, sequence_length, input_size)
            
        Returns:
            numpy.ndarray: 预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            
            X = X.to(self.device)
            predictions = self.model(X)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_future(self, recent_data, days=5):
        """
        预测未来几天的股价
        
        Args:
            recent_data: 最近的数据 (sequence_length, input_size)
            days: 预测天数
            
        Returns:
            numpy.ndarray: 预测的股价
        """
        # 确保输入数据形状正确
        if len(recent_data.shape) == 2:
            recent_data = recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])
        
        predictions = self.predict(recent_data)
        
        # 如果预测天数不匹配，进行调整
        if predictions.shape[1] != days:
            if days <= predictions.shape[1]:
                predictions = predictions[:, :days]
            else:
                # 如果需要更多天数，使用递归预测
                predictions = self._recursive_predict(recent_data, days)
        
        return predictions[0]  # 返回第一个样本的预测结果
    
    def _recursive_predict(self, data, days):
        """
        递归预测更多天数
        
        Args:
            data: 输入数据
            days: 预测天数
            
        Returns:
            numpy.ndarray: 预测结果
        """
        all_predictions = []
        current_data = data.copy()
        
        remaining_days = days
        while remaining_days > 0:
            # 预测当前批次
            batch_predictions = self.predict(current_data)
            batch_size = min(remaining_days, batch_predictions.shape[1])
            
            all_predictions.append(batch_predictions[0, :batch_size])
            remaining_days -= batch_size
            
            if remaining_days > 0:
                # 更新输入数据，使用预测值作为新的输入
                # 这里简化处理，实际应用中需要更复杂的特征工程
                new_features = np.zeros((1, batch_size, current_data.shape[2]))
                new_features[0, :, 3] = batch_predictions[0, :batch_size]  # 假设第4列是收盘价
                
                # 滑动窗口更新
                current_data = np.concatenate([current_data[:, batch_size:, :], new_features], axis=1)
        
        return np.concatenate(all_predictions).reshape(1, -1)
    
    def evaluate(self, test_data, stock_code):
        """
        评估模型性能

        Args:
            test_data: 测试数据 (X_test, y_test)
            stock_code: 股票代码

        Returns:
            dict: 评估指标
        """
        X_test, y_test = test_data

        print(f"📊 评估数据形状: X_test={X_test.shape}, y_test={y_test.shape}")

        # 进行预测
        predictions = self.predict(X_test)
        print(f"🔮 预测结果形状: {predictions.shape}")

        # 反标准化
        try:
            if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
                print("🔧 开始反标准化...")
                predictions_denorm = self.preprocessor.inverse_transform(predictions, target_column='close')
                y_test_denorm = self.preprocessor.inverse_transform(y_test, target_column='close')
                print(f"✅ 反标准化完成: pred={predictions_denorm.shape}, actual={y_test_denorm.shape}")
            else:
                print("⚠️ 缩放器不可用，使用原始数据")
                predictions_denorm = predictions
                y_test_denorm = y_test
        except Exception as e:
            print(f"❌ 反标准化失败: {str(e)}")
            print("⚠️ 使用标准化数据进行评估")
            predictions_denorm = predictions
            y_test_denorm = y_test

        # 保存预测结果
        self.predictions = predictions_denorm
        self.actual_values = y_test_denorm

        # 计算评估指标
        try:
            # 确保数据形状一致
            if predictions_denorm.shape != y_test_denorm.shape:
                print(f"⚠️ 调整数据形状: pred={predictions_denorm.shape}, actual={y_test_denorm.shape}")
                min_samples = min(predictions_denorm.shape[0], y_test_denorm.shape[0])
                if len(predictions_denorm.shape) > 1 and len(y_test_denorm.shape) > 1:
                    min_features = min(predictions_denorm.shape[1], y_test_denorm.shape[1])
                    predictions_denorm = predictions_denorm[:min_samples, :min_features]
                    y_test_denorm = y_test_denorm[:min_samples, :min_features]
                else:
                    predictions_denorm = predictions_denorm[:min_samples]
                    y_test_denorm = y_test_denorm[:min_samples]

            # 基本评估指标
            mse = np.mean((predictions_denorm - y_test_denorm) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions_denorm - y_test_denorm))

            # 计算MAPE（处理除零情况）
            y_test_nonzero = y_test_denorm.copy()
            y_test_nonzero[y_test_nonzero == 0] = 1e-8  # 避免除零
            mape = np.mean(np.abs((y_test_denorm - predictions_denorm) / y_test_nonzero)) * 100

            # 计算方向准确率（仅当有多个时间步时）
            if len(predictions_denorm.shape) > 1 and predictions_denorm.shape[1] > 1:
                pred_direction = np.sign(np.diff(predictions_denorm, axis=1))
                actual_direction = np.sign(np.diff(y_test_denorm, axis=1))
                direction_accuracy = np.mean(pred_direction == actual_direction) * 100
            else:
                # 单步预测的方向准确率
                if len(predictions_denorm) > 1:
                    pred_direction = np.sign(np.diff(predictions_denorm.flatten()))
                    actual_direction = np.sign(np.diff(y_test_denorm.flatten()))
                    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
                else:
                    direction_accuracy = 50.0  # 默认值

            metrics = {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'Direction_Accuracy': float(direction_accuracy)
            }

            print(f"\n{stock_code} 模型评估结果:")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"方向准确率: {direction_accuracy:.2f}%")

            return metrics

        except Exception as e:
            print(f"❌ 评估指标计算失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 返回默认指标
            return {
                'MSE': float('inf'),
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'MAPE': float('inf'),
                'Direction_Accuracy': 50.0
            }
    
    def plot_predictions(self, stock_code, num_samples=100):
        """
        绘制预测结果
        
        Args:
            stock_code: 股票代码
            num_samples: 显示的样本数量
        """
        if self.predictions is None or self.actual_values is None:
            print("没有预测结果可显示")
            return
        
        # 限制显示的样本数量
        num_samples = min(num_samples, len(self.predictions))
        predictions = self.predictions[:num_samples]
        actual_values = self.actual_values[:num_samples]
        
        # 创建时间轴
        time_steps = range(len(predictions))
        
        plt.figure(figsize=(15, 10))
        
        # 绘制每一天的预测
        for day in range(predictions.shape[1]):
            plt.subplot(2, 3, day + 1)
            plt.scatter(time_steps, actual_values[:, day], alpha=0.6, label='实际值', s=20)
            plt.scatter(time_steps, predictions[:, day], alpha=0.6, label='预测值', s=20)
            plt.title(f'第{day+1}天预测')
            plt.xlabel('样本')
            plt.ylabel('股价')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 整体对比图
        plt.subplot(2, 3, 6)
        plt.plot(actual_values.flatten(), label='实际值', alpha=0.7)
        plt.plot(predictions.flatten(), label='预测值', alpha=0.7)
        plt.title('整体预测对比')
        plt.xlabel('时间步')
        plt.ylabel('股价')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{stock_code} - 预测结果对比', fontsize=16)
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_predictions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_next_days(self, stock_data, stock_code, days=5):
        """
        预测接下来几天的股价

        Args:
            stock_data: 股票历史数据
            stock_code: 股票代码
            days: 预测天数

        Returns:
            dict: 预测结果
        """
        try:
            print(f"🔮 开始预测 {stock_code} 未来 {days} 天...")

            # 加载预处理器
            scaler_loaded = self.preprocessor.load_scaler(f'{stock_code}_scaler.pkl')
            if not scaler_loaded:
                print("⚠️ 无法加载预处理器，使用当前设置")

            # 预处理数据
            print("🔧 预处理数据...")
            df_with_indicators = self.preprocessor.add_technical_indicators(stock_data)
            feature_data = self.preprocessor.select_features(df_with_indicators)
            feature_data = feature_data.dropna()

            print(f"📊 特征数据形状: {feature_data.shape}")

            # 标准化
            if scaler_loaded:
                normalized_data = self.preprocessor.normalize_data(feature_data.values, fit_scaler=False)
            else:
                print("⚠️ 重新拟合缩放器")
                normalized_data = self.preprocessor.normalize_data(feature_data.values, fit_scaler=True)

            # 获取最近的序列数据
            sequence_length = getattr(self.preprocessor, 'sequence_length', DATA_CONFIG['sequence_length'])
            if len(normalized_data) < sequence_length:
                print(f"⚠️ 数据不足，调整序列长度: {sequence_length} -> {len(normalized_data)}")
                sequence_length = len(normalized_data)

            recent_sequence = normalized_data[-sequence_length:]
            print(f"📏 使用序列长度: {sequence_length}")

            # 进行预测
            predictions = self.predict_future(recent_sequence, days)
            print(f"🎯 预测形状: {predictions.shape}")

            # 反标准化
            try:
                if len(predictions.shape) == 1:
                    predictions_reshaped = predictions.reshape(-1, 1)
                else:
                    predictions_reshaped = predictions.reshape(-1, 1) if predictions.shape[1] == 1 else predictions

                predictions_denorm = self.preprocessor.inverse_transform(predictions_reshaped, target_column='close')

                if len(predictions_denorm.shape) > 1:
                    predictions_denorm = predictions_denorm.flatten()

                print(f"✅ 反标准化完成: {predictions_denorm.shape}")

            except Exception as e:
                print(f"❌ 反标准化失败: {str(e)}")
                print("⚠️ 使用原始预测值")
                predictions_denorm = predictions.flatten() if len(predictions.shape) > 1 else predictions

            # 创建预测日期
            last_date = stock_data.index[-1]
            pred_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions_denorm))]

            # 获取最后价格
            last_price = float(stock_data['close'].iloc[-1])

            # 构建结果
            result = {
                'dates': pred_dates,
                'predictions': predictions_denorm.tolist() if hasattr(predictions_denorm, 'tolist') else list(predictions_denorm),
                'last_price': last_price,
                'prediction_change': (predictions_denorm - last_price).tolist() if hasattr(predictions_denorm, 'tolist') else list(predictions_denorm - last_price)
            }

            print(f"✅ 预测完成，预测了 {len(predictions_denorm)} 天")
            return result

        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 返回默认结果
            last_price = float(stock_data['close'].iloc[-1])
            pred_dates = [stock_data.index[-1] + timedelta(days=i+1) for i in range(days)]

            return {
                'dates': pred_dates,
                'predictions': [last_price] * days,
                'last_price': last_price,
                'prediction_change': [0.0] * days
            }
