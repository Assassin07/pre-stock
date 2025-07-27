"""
使用示例
"""

from main import quick_predict
from data_fetcher import StockDataFetcher
from data_preprocessor import StockDataPreprocessor
from trainer import StockTrainer
from predictor import StockPredictor
from visualizer import StockVisualizer
from utils import create_directories


def example_1_quick_prediction():
    """示例1: 快速预测"""
    print("=" * 50)
    print("示例1: 快速预测平安银行(000001)未来5天走势")
    print("=" * 50)
    
    result = quick_predict('000001', days=5)
    if result:
        print("✅ 快速预测完成")
    else:
        print("❌ 快速预测失败")


def example_2_step_by_step():
    """示例2: 分步骤详细使用"""
    print("=" * 50)
    print("示例2: 分步骤预测招商银行(600036)")
    print("=" * 50)
    
    stock_code = "600036"
    
    # 创建目录
    create_directories()
    
    # 1. 获取数据
    print("1. 获取股票数据...")
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(stock_code)
    if df is None:
        print("❌ 无法获取数据")
        return
    
    print(f"✅ 获取到 {len(df)} 条数据")
    
    # 2. 数据预处理
    print("2. 数据预处理...")
    preprocessor = StockDataPreprocessor()
    train_data, val_data, test_data = preprocessor.prepare_data(df)
    input_size = len(preprocessor.feature_columns)
    print(f"✅ 特征数量: {input_size}")
    
    # 3. 可视化
    print("3. 数据可视化...")
    visualizer = StockVisualizer()
    df_with_indicators = preprocessor.add_technical_indicators(df)
    visualizer.plot_stock_data(df_with_indicators.tail(100), stock_code)
    
    # 4. 训练模型
    print("4. 训练LSTM模型...")
    trainer = StockTrainer('lstm', input_size, 5)
    trainer.train(train_data, val_data, stock_code)
    print("✅ 训练完成")
    
    # 5. 预测
    print("5. 进行预测...")
    predictor = StockPredictor('lstm', input_size, 5)
    predictor.load_model(stock_code)
    predictor.preprocessor = preprocessor
    
    # 评估模型
    metrics = predictor.evaluate(test_data, stock_code)
    
    # 预测未来
    future_prediction = predictor.predict_next_days(df, stock_code, 5)
    
    # 可视化预测结果
    visualizer.plot_future_prediction(df, future_prediction, stock_code)
    
    print("✅ 示例2完成")


def example_3_compare_models():
    """示例3: 比较不同模型"""
    print("=" * 50)
    print("示例3: 比较LSTM、GRU、Transformer模型")
    print("=" * 50)
    
    stock_code = "000002"  # 万科A
    models = ['lstm', 'gru', 'transformer']
    results = {}
    
    # 创建目录
    create_directories()
    
    # 获取数据
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(stock_code)
    if df is None:
        print("❌ 无法获取数据")
        return
    
    # 预处理
    preprocessor = StockDataPreprocessor()
    train_data, val_data, test_data = preprocessor.prepare_data(df)
    input_size = len(preprocessor.feature_columns)
    
    # 训练和评估不同模型
    for model_type in models:
        print(f"\n训练 {model_type.upper()} 模型...")
        
        # 训练
        trainer = StockTrainer(model_type, input_size, 5)
        trainer.train(train_data, val_data, f"{stock_code}_{model_type}")
        
        # 预测
        predictor = StockPredictor(model_type, input_size, 5)
        predictor.load_model(f"{stock_code}_{model_type}")
        predictor.preprocessor = preprocessor
        
        # 评估
        metrics = predictor.evaluate(test_data, f"{stock_code}_{model_type}")
        results[model_type] = metrics
    
    # 比较结果
    print("\n" + "=" * 50)
    print("模型比较结果:")
    print("=" * 50)
    print(f"{'模型':<12} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'方向准确率':<10}")
    print("-" * 50)
    
    for model_type, metrics in results.items():
        print(f"{model_type.upper():<12} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} "
              f"{metrics['MAPE']:<10.2f} {metrics['Direction_Accuracy']:<10.2f}")
    
    print("✅ 示例3完成")


def example_4_batch_prediction():
    """示例4: 批量预测多只股票"""
    print("=" * 50)
    print("示例4: 批量预测多只股票")
    print("=" * 50)
    
    # 选择几只热门股票
    stock_codes = ['000001', '000002', '600036', '600519']  # 平安银行、万科A、招商银行、贵州茅台
    stock_names = ['平安银行', '万科A', '招商银行', '贵州茅台']
    
    create_directories()
    
    results = {}
    
    for stock_code, stock_name in zip(stock_codes, stock_names):
        print(f"\n处理 {stock_name}({stock_code})...")
        
        try:
            # 获取数据
            fetcher = StockDataFetcher()
            df = fetcher.fetch_stock_data(stock_code)
            if df is None:
                print(f"❌ 无法获取 {stock_name} 数据")
                continue
            
            # 预处理
            preprocessor = StockDataPreprocessor()
            train_data, val_data, test_data = preprocessor.prepare_data(df)
            input_size = len(preprocessor.feature_columns)
            
            # 训练（使用较少的epoch以节省时间）
            trainer = StockTrainer('lstm', input_size, 3)
            # 减少训练轮数
            from config import TRAINING_CONFIG
            original_epochs = TRAINING_CONFIG['num_epochs']
            TRAINING_CONFIG['num_epochs'] = 20
            
            trainer.train(train_data, val_data, stock_code)
            
            # 恢复原始设置
            TRAINING_CONFIG['num_epochs'] = original_epochs
            
            # 预测
            predictor = StockPredictor('lstm', input_size, 3)
            predictor.load_model(stock_code)
            predictor.preprocessor = preprocessor
            
            future_prediction = predictor.predict_next_days(df, stock_code, 3)
            results[stock_name] = future_prediction
            
            print(f"✅ {stock_name} 处理完成")
            
        except Exception as e:
            print(f"❌ {stock_name} 处理失败: {str(e)}")
    
    # 显示所有结果
    print("\n" + "=" * 60)
    print("批量预测结果汇总:")
    print("=" * 60)
    
    for stock_name, prediction in results.items():
        print(f"\n{stock_name}:")
        print(f"当前价格: {prediction['last_price']:.2f}")
        for i, (date, price) in enumerate(zip(prediction['dates'], prediction['predictions'])):
            change = price - prediction['last_price']
            change_pct = change / prediction['last_price'] * 100
            print(f"  第{i+1}天: {price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
    
    print("✅ 示例4完成")


if __name__ == "__main__":
    print("🚀 A股股票预测系统使用示例")
    print("\n请选择要运行的示例:")
    print("1. 快速预测")
    print("2. 分步骤详细使用")
    print("3. 比较不同模型")
    print("4. 批量预测多只股票")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        example_1_quick_prediction()
    elif choice == '2':
        example_2_step_by_step()
    elif choice == '3':
        example_3_compare_models()
    elif choice == '4':
        example_4_batch_prediction()
    else:
        print("❌ 无效选择，运行默认示例")
        example_1_quick_prediction()
