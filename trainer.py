"""
模型训练模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import TRAINING_CONFIG, PATHS
from model import create_model


class StockTrainer:
    def __init__(self, model_type='lstm', input_size=20, output_size=5):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型
            input_size: 输入特征数量
            output_size: 输出大小
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = create_model(model_type, input_size, output_size)
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 创建模型保存目录
        os.makedirs(PATHS['model_dir'], exist_ok=True)
    
    def create_data_loader(self, X, y, batch_size, shuffle=True):
        """
        创建数据加载器
        
        Args:
            X: 输入数据
            y: 目标数据
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader: 数据加载器
        """
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
        
        return data_loader
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in tqdm(train_loader, desc="训练中"):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 检查和调整张量维度
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)  # 添加维度 [batch_size] -> [batch_size, 1]

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            # 确保输出和目标维度匹配
            if outputs.shape != batch_y.shape:
                print(f"⚠️ 维度不匹配: outputs={outputs.shape}, targets={batch_y.shape}")
                # 如果输出维度大于目标维度，截取
                if outputs.shape[1] > batch_y.shape[1]:
                    outputs = outputs[:, :batch_y.shape[1]]
                # 如果目标维度大于输出维度，截取目标
                elif batch_y.shape[1] > outputs.shape[1]:
                    batch_y = batch_y[:, :outputs.shape[1]]

            loss = self.criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """
        验证一个epoch

        Args:
            val_loader: 验证数据加载器

        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 检查和调整张量维度
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)

                outputs = self.model(batch_X)

                # 确保输出和目标维度匹配
                if outputs.shape != batch_y.shape:
                    if outputs.shape[1] > batch_y.shape[1]:
                        outputs = outputs[:, :batch_y.shape[1]]
                    elif batch_y.shape[1] > outputs.shape[1]:
                        batch_y = batch_y[:, :outputs.shape[1]]

                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def train(self, train_data, val_data, stock_code):
        """
        训练模型
        
        Args:
            train_data: 训练数据 (X_train, y_train)
            val_data: 验证数据 (X_val, y_val)
            stock_code: 股票代码
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 创建数据加载器
        train_loader = self.create_data_loader(
            X_train, y_train, TRAINING_CONFIG['batch_size'], shuffle=True
        )
        val_loader = self.create_data_loader(
            X_val, y_val, TRAINING_CONFIG['batch_size'], shuffle=False
        )
        
        print(f"开始训练模型，股票代码: {stock_code}")
        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        for epoch in range(TRAINING_CONFIG['num_epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印进度
            print(f"Epoch [{epoch+1}/{TRAINING_CONFIG['num_epochs']}] - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                self.save_model(stock_code, epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= TRAINING_CONFIG['patience']:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
        
        print("训练完成！")
        return self.train_losses, self.val_losses
    
    def save_model(self, stock_code, epoch, val_loss, is_best=False):
        """
        保存模型
        
        Args:
            stock_code: 股票代码
            epoch: 当前轮次
            val_loss: 验证损失
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            filename = f"{stock_code}_best_model.pth"
        else:
            filename = f"{stock_code}_model_epoch_{epoch}.pth"
        
        filepath = os.path.join(PATHS['model_dir'], filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"最佳模型已保存: {filepath}")
    
    def load_model(self, stock_code, is_best=True):
        """
        加载模型
        
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
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            print(f"模型已加载: {filepath}")
            return True
        else:
            print(f"模型文件不存在: {filepath}")
            return False
    
    def plot_training_history(self, stock_code):
        """
        绘制训练历史
        
        Args:
            stock_code: 股票代码
        """
        if not self.train_losses or not self.val_losses:
            print("没有训练历史数据")
            return
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title(f'{stock_code} - 训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.title(f'{stock_code} - 训练历史 (对数尺度)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        plt.savefig(os.path.join(PATHS['results_dir'], f'{stock_code}_training_history.png'))
        plt.show()
