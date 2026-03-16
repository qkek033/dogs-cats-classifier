import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """모델 평가 클래스"""
    
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.predictions = []
        self.targets = []
        self.scores = []
    
    def update(self, outputs, targets):
        """배치 업데이트"""
        with torch.no_grad():
            # 확률 계산
            if isinstance(outputs, torch.Tensor):
                probs = torch.softmax(outputs, dim=1)
                scores = probs[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs
                scores = None
            
            targets = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            
            self.predictions.extend(preds)
            self.targets.extend(targets)
            if scores is not None:
                self.scores.extend(scores)
    
    def compute_metrics(self):
        """메트릭 계산"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(targets, predictions)
        }
        
        # ROC-AUC (이진 분류)
        if self.num_classes == 2 and len(self.scores) > 0:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, self.scores)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """혼동 행렬 시각화"""
        metrics = self.compute_metrics()
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        return cm
    
    def print_report(self):
        """리포트 출력"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("="*50 + "\n")
        
        return metrics
