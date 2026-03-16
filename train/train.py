import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import yaml
import logging
import mlflow
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from dataset import get_dataloaders, get_transforms
from model import create_model, count_parameters
from evaluate import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """모델 훈련 클래스"""
    
    def __init__(self, config_path='../config/training_config.yaml', device='cuda'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.best_metric = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def setup_mlflow(self):
        """MLflow 설정"""
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['backend_store_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            'epochs': self.config['training']['epochs'],
            'batch_size': self.config['training']['batch_size'],
            'learning_rate': self.config['training']['learning_rate'],
            'optimizer': self.config['optimizer']['name'],
            'scheduler': self.config['scheduler']['name']
        })
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """한 에포크 훈련"""
        model.train()
        total_loss = 0
        evaluator = Evaluator()
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            evaluator.update(outputs, labels)
            pbar.set_postfix({'loss': loss.item()})
        
        metrics = evaluator.compute_metrics()
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, metrics['accuracy']
    
    def validate(self, model, val_loader, criterion):
        """검증"""
        model.eval()
        total_loss = 0
        evaluator = Evaluator()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                evaluator.update(outputs, labels)
        
        metrics = evaluator.compute_metrics()
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def train(self, model_config='../config/config.yaml', data_dir='../data/raw'):
        """완전한 훈련 루프"""
        
        # 설정 로드
        with open(model_config) as f:
            model_cfg = yaml.safe_load(f)
        
        # 모델 생성
        model = create_model(model_cfg, device=self.device)
        logger.info(f"Model parameters: {count_parameters(model):,}")
        
        # 데이터 로더
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir=data_dir,
            batch_size=self.config['training']['batch_size']
        )
        logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        
        # 손실 함수
        criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['optimizer']['betas']
        )
        
        # 스케줄러
        scheduler_name = self.config['scheduler']['name']
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['t_max'],
                eta_min=self.config['scheduler']['eta_min']
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        # 조기 종료 설정
        early_stop_config = self.config.get('early_stopping', {})
        early_stop_enabled = early_stop_config.get('enabled', False)
        early_stop_patience = early_stop_config.get('patience', 5)
        
        # 체크포인트 디렉토리
        checkpoint_dir = Path(self.config['checkpoint']['model_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow 시작
        self.setup_mlflow()
        
        with mlflow.start_run():
            for epoch in range(self.config['training']['epochs']):
                logger.info(f"\nEpoch [{epoch+1}/{self.config['training']['epochs']}]")
                
                # 훈련
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                
                # 검증
                val_loss, val_metrics = self.validate(model, val_loader, criterion)
                val_acc = val_metrics['accuracy']
                
                # 학습률 업데이트
                scheduler.step()
                
                # 로깅
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # MLflow 로깅
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1']
                }, step=epoch)
                
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # 최고 성능 모델 저장
                if val_acc > self.best_metric:
                    self.best_metric = val_acc
                    self.patience_counter = 0
                    
                    checkpoint_path = checkpoint_dir / f"best_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': model_cfg
                    }, checkpoint_path)
                    logger.info(f"Best model saved: {checkpoint_path}")
                    mlflow.pytorch.log_model(model, "best_model")
                else:
                    self.patience_counter += 1
                
                # 조기 종료
                if early_stop_enabled and self.patience_counter >= early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 최종 메트릭
            logger.info(f"\nBest Validation Accuracy: {self.best_metric:.4f}")
            mlflow.log_metric("best_val_acc", self.best_metric)
        
        return model, self.history


if __name__ == '__main__':
    trainer = Trainer(device='cuda' if torch.cuda.is_available() else 'cpu')
    model, history = trainer.train(
        model_config='../config/config.yaml',
        data_dir='../data/raw'
    )
