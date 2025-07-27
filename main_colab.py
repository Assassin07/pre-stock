"""
Google Colab专用主程序
针对Colab环境优化的股票预测系统
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from datetime import datetime

# 检测是否在Colab环境中运行
try:
    import google.colab
    IN_COLAB = True
    print("🌟 检测到Google Colab环境")
except ImportError:
    IN_COLAB = False
    print("💻 本地环境")

# 根据环境选择配置文件
if IN_COLAB:
    try:
        from config_colab import *
        print("✅ 使用Colab优化配置")
    except ImportError:
        from config import *
        print("⚠️ 使用默认配置")
else:
    from config import *

from data_fetcher import StockDataFetcher
from data_preprocessor import StockDataPreprocessor
from trainer import StockTrainer
from predictor import StockPredictor
from visualizer import StockVisualizer
from utils import create_directories, setup_logging, calculate_technical_signals


def colab_quick_predict(stock_code, days=3, model_type='lstm', mode='quick'):
    """
    Colab专用快速预测函数
    
    Args:
        stock_code: 股票代码
        days: 预测天数
        model_type: 模型类型
        mode: 运行模式 ('quick', 'normal', 'performance')
    """
    print(f"🚀 Colab快速预测 {stock_code} 未来 {days} 天走势")
    print(f"🤖 使用模型: {model_type.upper()}")
    print(f"⚡ 运行模式: {mode}")
    
    # 根据模式调整配置
    if mode == 'quick':
        config = QUICK_MODE_CONFIG
        print("🏃 快速模式 - 适合演示和测试")
    elif mode == 'performance':
        config = PERFORMANCE_MODE_CONFIG
        print("🏆 性能模式 - 需要更多GPU资源")
    else:
        config = DATA_CONFIG
        print("⚖️ 标准模式 - 平衡速度和精度")
    
    # 创建目录
    create_directories()
    
    try:
        # 1. 获取数据
        print("\n📊 步骤1: 获取股票数据")
        fetcher = StockDataFetcher()
        df = fetcher.fetch_stock_data(stock_code)
        if df is None:
            print("❌ 无法获取股票数据")
            return None
        
        # 限制数据量以节省内存
        if IN_COLAB and len(df) > COLAB_CONFIG.get('max_data_points', 1000):
            df = df.tail(COLAB_CONFIG['max_data_points'])
            print(f"⚠️ 数据量限制为 {len(df)} 条以节省内存")
        
        print(f"✅ 获取到 {len(df)} 条数据")
        
        # 2. 数据预处理
        print("\n🔧 步骤2: 数据预处理")
        preprocessor = StockDataPreprocessor()
        
        # 使用配置中的参数
        preprocessor.sequence_length = config.get('sequence_length', 30)
        preprocessor.prediction_days = days
        
        train_data, val_data, test_data = preprocessor.prepare_data(df)
        input_size = len(preprocessor.feature_columns)
        
        print(f"✅ 特征数量: {input_size}")
        print(f"📏 序列长度: {preprocessor.sequence_length}")
        
        # 3. 模型训练
        print(f"\n🤖 步骤3: 训练{model_type.upper()}模型")
        
        # 使用配置中的模型参数
        model_config = {
            'input_size': input_size,
            'hidden_size': config.get('hidden_size', 64),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.2),
            'bidirectional': config.get('bidirectional', True)
        }
        
        trainer = StockTrainer(model_type, **model_config)
        
        # 使用配置中的训练参数
        original_config = TRAINING_CONFIG.copy()
        TRAINING_CONFIG.update({
            'batch_size': config.get('batch_size', 16),
            'num_epochs': config.get('num_epochs', 30),
            'patience': config.get('patience', 5),
            'learning_rate': config.get('learning_rate', 0.002)
        })
        
        # 训练模型
        train_losses, val_losses = trainer.train(train_data, val_data, stock_code)
        
        # 恢复原始配置
        TRAINING_CONFIG.update(original_config)
        
        print("✅ 模型训练完成")
        
        # 4. 预测
        print("\n🔮 步骤4: 进行预测")
        predictor = StockPredictor(model_type, input_size, days)
        predictor.load_model(stock_code)
        predictor.preprocessor = preprocessor
        
        # 预测未来
        future_prediction = predictor.predict_next_days(df, stock_code, days)
        
        # 5. 结果展示
        print("\n📈 预测结果:")
        print("=" * 50)
        print(f"当前价格: {future_prediction['last_price']:.2f}")
        print("-" * 50)
        
        for i, (date, price) in enumerate(zip(
            future_prediction['dates'], 
            future_prediction['predictions']
        )):
            change = price - future_prediction['last_price']
            change_pct = change / future_prediction['last_price'] * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"第{i+1}天 ({date.strftime('%Y-%m-%d')}): "
                  f"{price:.2f} ({change:+.2f}, {change_pct:+.2f}%) {direction}")
        
        # 总体趋势分析
        total_change = future_prediction['predictions'][-1] - future_prediction['last_price']
        total_change_pct = total_change / future_prediction['last_price'] * 100
        
        print("\n📊 总体趋势分析:")
        if total_change_pct > 2:
            print(f"🟢 看涨 (+{total_change_pct:.2f}%)")
        elif total_change_pct < -2:
            print(f"🔴 看跌 ({total_change_pct:.2f}%)")
        else:
            print(f"🟡 震荡 ({total_change_pct:+.2f}%)")
        
        # 6. 技术分析信号
        print("\n📊 技术分析信号:")
        df_with_indicators = preprocessor.add_technical_indicators(df)
        signals = calculate_technical_signals(df_with_indicators)
        for indicator, signal in signals.items():
            emoji = "🟢" if "涨" in signal else "🔴" if "跌" in signal else "🟡"
            print(f"{indicator.upper()}: {signal} {emoji}")
        
        print("\n🎉 预测完成！")
        
        # 如果在Colab环境中，自动保存结果
        if IN_COLAB and COLAB_CONFIG.get('auto_download_results', False):
            try:
                from google.colab import files
                import json
                
                # 保存预测结果为JSON
                result_data = {
                    'stock_code': stock_code,
                    'prediction_date': datetime.now().isoformat(),
                    'current_price': float(future_prediction['last_price']),
                    'predictions': [
                        {
                            'date': date.isoformat(),
                            'price': float(price),
                            'change': float(price - future_prediction['last_price']),
                            'change_pct': float((price - future_prediction['last_price']) / future_prediction['last_price'] * 100)
                        }
                        for date, price in zip(future_prediction['dates'], future_prediction['predictions'])
                    ],
                    'technical_signals': signals,
                    'model_type': model_type,
                    'mode': mode
                }
                
                filename = f"{stock_code}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
                
                print(f"\n💾 预测结果已保存: {filename}")
                
            except Exception as e:
                print(f"⚠️ 结果保存失败: {str(e)}")
        
        return future_prediction
        
    except Exception as e:
        print(f"\n❌ 预测过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def colab_batch_predict(stock_codes, days=3, model_type='lstm'):
    """
    Colab批量预测
    
    Args:
        stock_codes: 股票代码列表
        days: 预测天数
        model_type: 模型类型
    """
    print(f"📊 开始批量预测 {len(stock_codes)} 只股票")
    
    results = {}
    
    for i, stock_code in enumerate(stock_codes, 1):
        print(f"\n[{i}/{len(stock_codes)}] 处理 {stock_code}...")
        
        try:
            result = colab_quick_predict(stock_code, days, model_type, mode='quick')
            if result:
                results[stock_code] = result
                print(f"✅ {stock_code} 完成")
            else:
                print(f"❌ {stock_code} 失败")
        except Exception as e:
            print(f"❌ {stock_code} 出错: {str(e)}")
    
    # 汇总结果
    if results:
        print("\n📈 批量预测结果汇总:")
        print("=" * 60)
        
        for stock_code, result in results.items():
            total_change_pct = (result['predictions'][-1] - result['last_price']) / result['last_price'] * 100
            trend = "📈" if total_change_pct > 0 else "📉"
            print(f"{stock_code}: {result['last_price']:.2f} → {result['predictions'][-1]:.2f} "
                  f"({total_change_pct:+.2f}%) {trend}")
    
    return results


def main():
    """Colab主函数"""
    print("🚀 A股股票预测系统 - Google Colab版")
    print("=" * 50)
    
    # 示例使用
    stock_code = "000001"  # 平安银行
    days = 3
    model_type = "lstm"
    
    print(f"📊 示例预测: {stock_code}")
    print(f"🔮 预测天数: {days}")
    print(f"🤖 模型类型: {model_type}")
    
    result = colab_quick_predict(stock_code, days, model_type, mode='quick')
    
    if result:
        print("\n✅ 示例运行成功！")
        print("💡 你可以修改股票代码和参数来预测其他股票")
    else:
        print("\n❌ 示例运行失败")


if __name__ == "__main__":
    main()
