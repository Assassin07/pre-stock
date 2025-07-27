"""
A股股票预测系统主程序
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import StockDataFetcher
from data_preprocessor import StockDataPreprocessor
from trainer import StockTrainer
from predictor import StockPredictor
from visualizer import StockVisualizer
from utils import create_directories, setup_logging, calculate_technical_signals
from config import DEFAULT_STOCK_CODE, DATA_CONFIG


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股股票预测系统')
    parser.add_argument('--stock_code', type=str, default=DEFAULT_STOCK_CODE,
                       help='股票代码 (默认: 000001)')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'both'], default='both',
                       help='运行模式: train(训练), predict(预测), both(训练+预测)')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'gru', 'transformer'], default='lstm',
                       help='模型类型 (默认: lstm)')
    parser.add_argument('--days', type=int, default=5,
                       help='预测天数 (默认: 5)')
    parser.add_argument('--start_date', type=str, default=None,
                       help='数据开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='数据结束日期 (格式: YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 设置日志和创建目录
    setup_logging()
    create_directories()
    
    print("=" * 60)
    print("🚀 A股股票预测系统")
    print("=" * 60)
    print(f"股票代码: {args.stock_code}")
    print(f"运行模式: {args.mode}")
    print(f"模型类型: {args.model_type}")
    print(f"预测天数: {args.days}")
    print("=" * 60)
    
    try:
        # 1. 数据获取
        print("\n📊 步骤1: 获取股票数据")
        fetcher = StockDataFetcher()
        
        # 尝试从本地加载数据
        df = fetcher.load_data(args.stock_code)
        if df is None or len(df) < 100:
            print("本地数据不存在或数据量不足，从网络获取...")
            df = fetcher.fetch_stock_data(
                args.stock_code, 
                start_date=args.start_date, 
                end_date=args.end_date
            )
            if df is None:
                print("❌ 无法获取股票数据，程序退出")
                return
            fetcher.save_data(df, args.stock_code)
        
        print(f"✅ 数据获取完成，共 {len(df)} 条记录")
        print(f"数据时间范围: {df.index[0].date()} 到 {df.index[-1].date()}")
        
        # 2. 数据预处理
        print("\n🔧 步骤2: 数据预处理")
        preprocessor = StockDataPreprocessor()
        train_data, val_data, test_data = preprocessor.prepare_data(df)
        
        # 保存预处理器
        preprocessor.save_scaler(f'{args.stock_code}_scaler.pkl')
        
        input_size = len(preprocessor.feature_columns)
        print(f"✅ 数据预处理完成，特征数量: {input_size}")
        
        # 3. 可视化原始数据
        print("\n📈 步骤3: 数据可视化")
        visualizer = StockVisualizer()
        
        # 添加技术指标用于可视化
        df_with_indicators = preprocessor.add_technical_indicators(df)
        visualizer.plot_stock_data(df_with_indicators.tail(200), args.stock_code)
        visualizer.plot_technical_indicators(df_with_indicators.tail(200), args.stock_code)
        
        # 4. 模型训练
        if args.mode in ['train', 'both']:
            print(f"\n🤖 步骤4: 训练{args.model_type.upper()}模型")
            trainer = StockTrainer(args.model_type, input_size, args.days)
            
            # 训练模型
            train_losses, val_losses = trainer.train(train_data, val_data, args.stock_code)
            
            # 绘制训练历史
            trainer.plot_training_history(args.stock_code)
            print("✅ 模型训练完成")
        
        # 5. 模型预测和评估
        if args.mode in ['predict', 'both']:
            print(f"\n🔮 步骤5: 模型预测")
            predictor = StockPredictor(args.model_type, input_size, args.days)
            
            # 加载训练好的模型
            if not predictor.load_model(args.stock_code):
                print("❌ 无法加载训练好的模型，请先运行训练模式")
                return
            
            # 加载预处理器
            predictor.preprocessor.load_scaler(f'{args.stock_code}_scaler.pkl')
            
            # 在测试集上评估
            print("\n📊 测试集评估:")
            metrics = predictor.evaluate(test_data, args.stock_code)
            
            # 绘制预测结果
            predictor.plot_predictions(args.stock_code)
            
            # 预测未来几天
            print(f"\n🔮 预测未来{args.days}天:")
            future_prediction = predictor.predict_next_days(df, args.stock_code, args.days)
            
            # 显示预测结果
            print("\n预测结果:")
            for i, (date, price, change) in enumerate(zip(
                future_prediction['dates'], 
                future_prediction['predictions'],
                future_prediction['prediction_change']
            )):
                print(f"第{i+1}天 ({date.strftime('%Y-%m-%d')}): "
                      f"{price:.2f} ({change:+.2f}, {change/future_prediction['last_price']*100:+.2f}%)")
            
            # 可视化未来预测
            visualizer.plot_future_prediction(df, future_prediction, args.stock_code)
            
            # 技术分析信号
            print("\n📊 技术分析信号:")
            signals = calculate_technical_signals(df_with_indicators)
            for indicator, signal in signals.items():
                print(f"{indicator.upper()}: {signal}")
            
            print("✅ 预测完成")
        
        print("\n🎉 程序执行完成！")
        print("📁 结果文件保存在 results/ 目录中")
        print("🤖 模型文件保存在 models/ 目录中")
        
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


def quick_predict(stock_code, days=5):
    """
    快速预测函数（用于简单调用）
    
    Args:
        stock_code: 股票代码
        days: 预测天数
    """
    print(f"🚀 快速预测 {stock_code} 未来 {days} 天走势")
    
    # 创建目录
    create_directories()
    
    # 获取数据
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(stock_code)
    if df is None:
        print("❌ 无法获取股票数据")
        return None
    
    # 预处理
    preprocessor = StockDataPreprocessor()
    train_data, val_data, test_data = preprocessor.prepare_data(df)
    input_size = len(preprocessor.feature_columns)
    
    # 训练模型
    trainer = StockTrainer('lstm', input_size, days)
    trainer.train(train_data, val_data, stock_code)
    
    # 预测
    predictor = StockPredictor('lstm', input_size, days)
    predictor.load_model(stock_code)
    predictor.preprocessor = preprocessor
    
    future_prediction = predictor.predict_next_days(df, stock_code, days)
    
    # 显示结果
    print("\n预测结果:")
    for i, (date, price) in enumerate(zip(future_prediction['dates'], future_prediction['predictions'])):
        print(f"第{i+1}天 ({date.strftime('%Y-%m-%d')}): {price:.2f}")
    
    return future_prediction


if __name__ == "__main__":
    main()
