"""
导航预测模型模块

包含：
1. RiskPredictionModel - 风险预测模型（基于GradientBoosting）
2. PassabilityProbabilityModel - 可达性概率模型（基于GradientBoosting）
3. MultiTaskNavigationModel - 多任务深度神经网络模型（PyTorch）
   同时预测风险评分和可达性概率
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，多任务DNN模型不可用")


class RiskPredictionModel:
    """
    风险预测模型 - 基于机器学习替代规则计算
    
    输入特征：
    - 边特征：水深、宽度、水域类型、样本量、航速等
    - 船舶特征：吃水、宽度、高度、吨位、船型
    
    输出：0-100 风险评分
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or "output/risk_prediction_model.pkl"
        self.is_trained = False
        
    def _extract_features(self, edge_features: Dict, ship_features: Dict) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 边特征
        features.append(edge_features.get('avg_distance', 100))
        features.append(edge_features.get('avg_travel_time', 30))
        features.append(edge_features.get('avg_actual_speed', 5))
        features.append(edge_features.get('segment_count', 0))
        features.append(edge_features.get('speed_reliability', 0.8))
        features.append(edge_features.get('node_degree_from', 0))
        features.append(edge_features.get('node_degree_to', 0))
        features.append(edge_features.get('edge_betweenness', 0))
        features.append(1 if edge_features.get('waterway_type') == 'narrow' else 0)
        
        # 船舶特征
        features.append(ship_features.get('draft', 5))
        features.append(ship_features.get('width', 15))
        features.append(ship_features.get('height', 20))
        features.append(ship_features.get('length', 100))
        features.append(ship_features.get('tonnage', 5000))
        features.append(ship_features.get('max_speed', 15))
        
        # 交互特征（重要！）
        draft_margin = edge_features.get('min_depth', 15) - ship_features.get('draft', 5)
        width_margin = edge_features.get('max_width', 100) - ship_features.get('width', 15)
        features.append(draft_margin)
        features.append(width_margin)
        features.append(draft_margin * width_margin)  # 联合裕度
        
        return np.array(features).reshape(1, -1)
    
    def generate_pseudo_labels(self, edge_features_dict: Dict, 
                               constraint_checker,
                               ship_templates: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于当前规则生成伪标签用于训练
        
        Returns:
            X: 特征矩阵
            y: 风险评分（规则计算结果）
        """
        from ship_navigator import ShipCharacteristics
        
        X_list = []
        y_list = []
        
        for edge_key, edge_feat in edge_features_dict.items():
            for ship_template in ship_templates:
                # 提取特征
                features = self._extract_features(edge_feat, ship_template)
                X_list.append(features[0])
                
                # 创建 ShipCharacteristics 对象用于规则计算
                ship_obj = ShipCharacteristics(
                    ship_name="template",
                    draft=ship_template.get('draft', 5),
                    width=ship_template.get('width', 15),
                    height=ship_template.get('height', 20),
                    length=ship_template.get('length', 100),
                    tonnage=ship_template.get('tonnage', 5000),
                    max_speed=ship_template.get('max_speed', 15)
                )
                
                # 用规则计算伪标签
                risk = constraint_checker._rule_based_risk_score(edge_key, ship_obj)
                y_list.append(risk)
        
        return np.array(X_list), np.array(y_list)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        logger.info("训练风险预测模型，样本数: %d", len(X))
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练梯度提升回归器
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # 评估
        train_score = self.model.score(X_scaled, y)
        logger.info("模型训练完成，R²得分: %.4f", train_score)
        
        # 保存
        self.save()
    
    def predict(self, edge_features: Dict, ship_features: Dict) -> float:
        """预测风险评分"""
        if not self.is_trained:
            logger.warning("风险模型未训练，返回默认风险50")
            return 50.0
        
        features = self._extract_features(edge_features, ship_features)
        features_scaled = self.scaler.transform(features)
        risk = self.model.predict(features_scaled)[0]
        
        # 限制在0-100范围
        return float(np.clip(risk, 0, 100))
    
    def save(self):
        """保存模型"""
        if self.model is not None:
            data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info("风险预测模型已保存: %s", self.model_path)
    
    def load(self) -> bool:
        """加载模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            logger.info("风险预测模型已加载: %s", self.model_path)
            return True
        return False


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    print("=" * 60)
    print("导航预测模型模块 - 独立测试")
    print("=" * 60)
    
    print("\n可用模型类:")
    print("  1. RiskPredictionModel - 风险预测模型")
    print("  2. PassabilityProbabilityModel - 可达性概率模型")
    print("  3. MultiTaskNavigationModel - 多任务DNN模型 (需要PyTorch)")
    
    print("\n检查依赖...")
    print(f"  PyTorch: {'已安装' if TORCH_AVAILABLE else '未安装'}")
    
    print("\n测试模型初始化...")
    
    risk_model = RiskPredictionModel()
    print(f"  RiskPredictionModel: OK (model_path={risk_model.model_path})")
    
    pass_model = PassabilityProbabilityModel()
    print(f"  PassabilityProbabilityModel: OK (model_path={pass_model.model_path})")
    
    if TORCH_AVAILABLE:
        multi_model = MultiTaskNavigationModel()
        print(f"  MultiTaskNavigationModel: OK (model_path={multi_model.model_path})")
    
    print("\n测试特征提取...")
    test_edge = {
        'avg_distance': 1500,
        'avg_travel_time': 600,
        'avg_actual_speed': 5.0,
        'segment_count': 10,
        'speed_reliability': 0.85,
        'node_degree_from': 3,
        'node_degree_to': 2,
        'edge_betweenness': 0.1,
        'waterway_type': 'open',
        'min_depth': 20,
        'max_width': 150,
        'max_height': 50
    }
    test_ship = {
        'draft': 6,
        'width': 20,
        'height': 25,
        'length': 120,
        'tonnage': 8000,
        'max_speed': 12
    }
    
    features = risk_model._extract_features(test_edge, test_ship)
    print(f"  特征向量维度: {features.shape[1]}")
    print(f"  特征值: {features[0][:5]}... (前5个)")
    
    print("\n" + "=" * 60)
    print("测试完成！模型初始化正常。")
    print("=" * 60)


class PassabilityProbabilityModel:
    """
    可达性概率模型 - 概率化判断船舶能否通过某边
    
    替代二元判断（能/不能），输出0-1概率
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or "output/passability_model.pkl"
        self.is_trained = False
    
    def _extract_features(self, edge_features: Dict, ship_features: Dict) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 边特征
        features.append(edge_features.get('avg_distance', 100))
        features.append(edge_features.get('avg_actual_speed', 5))
        features.append(edge_features.get('segment_count', 0))
        features.append(edge_features.get('speed_reliability', 0.8))
        features.append(1 if edge_features.get('waterway_type') == 'narrow' else 0)
        
        # 船舶特征
        features.append(ship_features.get('draft', 5))
        features.append(ship_features.get('width', 15))
        features.append(ship_features.get('height', 20))
        features.append(ship_features.get('length', 100))
        
        # 关键：裕度特征
        draft_ratio = ship_features.get('draft', 5) / max(edge_features.get('min_depth', 15), 0.1)
        width_ratio = ship_features.get('width', 15) / max(edge_features.get('max_width', 100), 0.1)
        height_ratio = ship_features.get('height', 20) / max(edge_features.get('max_height', 100), 0.1)
        
        features.append(draft_ratio)
        features.append(width_ratio)
        features.append(height_ratio)
        features.append(max(draft_ratio, width_ratio, height_ratio))  # 最严格约束
        
        return np.array(features).reshape(1, -1)
    
    def generate_training_data(self, edge_features_dict: Dict,
                               constraint_checker,
                               ship_templates: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据
        
        标签：1=可通过，0=不可通过（基于当前规则）
        """
        from ship_navigator import ShipCharacteristics
        
        X_list = []
        y_list = []
        
        for edge_key, edge_feat in edge_features_dict.items():
            for ship_template in ship_templates:
                features = self._extract_features(edge_feat, ship_template)
                X_list.append(features[0])
                
                # 创建 ShipCharacteristics 对象
                ship_obj = ShipCharacteristics(
                    ship_name="template",
                    draft=ship_template.get('draft', 5),
                    width=ship_template.get('width', 15),
                    height=ship_template.get('height', 20),
                    length=ship_template.get('length', 100),
                    tonnage=ship_template.get('tonnage', 5000),
                    max_speed=ship_template.get('max_speed', 15)
                )
                
                # 规则判断标签
                passable, _ = constraint_checker.check_edge_passable(edge_key, ship_obj)
                y_list.append(1 if passable else 0)
        
        return np.array(X_list), np.array(y_list)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        logger.info("训练可达性概率模型，样本数: %d", len(X))
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练梯度提升分类器（输出概率）
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # 评估
        train_score = self.model.score(X_scaled, y)
        logger.info("模型训练完成，准确率: %.4f", train_score)
        
        self.save()
    
    def predict_proba(self, edge_features: Dict, ship_features: Dict) -> float:
        """预测可通过概率"""
        if not self.is_trained:
            logger.warning("可达性模型未训练，返回默认概率0.5")
            return 0.5
        
        features = self._extract_features(edge_features, ship_features)
        features_scaled = self.scaler.transform(features)
        
        # 返回正类（可通过）的概率
        proba = self.model.predict_proba(features_scaled)[0][1]
        return float(proba)
    
    def save(self):
        """保存模型"""
        if self.model is not None:
            data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info("可达性概率模型已保存: %s", self.model_path)
    
    def load(self) -> bool:
        """加载模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            logger.info("可达性概率模型已加载: %s", self.model_path)
            return True
        return False


# ==================== 多任务深度神经网络模型 ====================

class MultiTaskDNN(nn.Module):
    """
    多任务深度神经网络（改进版）
    
    基于文献改进：
    - DIR (NeurIPS 2021): 简化架构避免过拟合，3层共享层
    - UVOTE (GCPR 2024): NLL损失替代MSE，预测均值+方差处理不平衡回归
    - 风险回归头输出 (mu, log_var)，用高斯NLL训练
    - 可达性分类头输出logit，用BCE with logits训练
    """
    
    def __init__(self, input_dim=20, hidden_dim=128, dropout_rate=0.3):
        super(MultiTaskDNN, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
        )
        
        self.risk_mu = nn.Linear(64, 1)
        self.risk_log_var = nn.Linear(64, 1)
        
        self.passability_head = nn.Linear(64, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h = self.shared(x)
        risk_mu = torch.sigmoid(self.risk_mu(h)) * 100
        risk_log_var = self.risk_log_var(h)
        passability_logit = self.passability_head(h)
        return risk_mu, risk_log_var, passability_logit


class MultiTaskNavigationModel:
    """
    多任务导航预测模型（PyTorch实现）
    
    同时训练风险预测和可达性概率两个任务
    """
    
    def __init__(self, model_path: str = None, input_dim: int = 20):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用多任务DNN模型")
        
        self.model = MultiTaskDNN(input_dim=input_dim)
        self.scaler = StandardScaler()
        self.model_path = model_path or "output/multitask_navigation_model.pt"
        self.scaler_path = model_path.replace('.pt', '_scaler.pkl') if model_path else "output/multitask_navigation_scaler.pkl"
        self.is_trained = False
        self.input_dim = input_dim
        
    def _extract_features(self, edge_features: Dict, ship_features: Dict) -> np.ndarray:
        """提取特征向量（与RiskPredictionModel相同）"""
        features = []
        
        # 边特征
        features.append(edge_features.get('avg_distance', 100))
        features.append(edge_features.get('avg_travel_time', 30))
        features.append(edge_features.get('avg_actual_speed', 5))
        features.append(edge_features.get('segment_count', 0))
        features.append(edge_features.get('speed_reliability', 0.8))
        features.append(edge_features.get('node_degree_from', 0))
        features.append(edge_features.get('node_degree_to', 0))
        features.append(edge_features.get('edge_betweenness', 0))
        features.append(1 if edge_features.get('waterway_type') == 'narrow' else 0)
        
        # 船舶特征
        features.append(ship_features.get('draft', 5))
        features.append(ship_features.get('width', 15))
        features.append(ship_features.get('height', 20))
        features.append(ship_features.get('length', 100))
        features.append(ship_features.get('tonnage', 5000))
        features.append(ship_features.get('max_speed', 15))
        
        # 交互特征
        draft_margin = edge_features.get('min_depth', 15) - ship_features.get('draft', 5)
        width_margin = edge_features.get('max_width', 100) - ship_features.get('width', 15)
        height_margin = edge_features.get('max_height', 100) - ship_features.get('height', 20)
        features.append(draft_margin)
        features.append(width_margin)
        features.append(height_margin)
        features.append(draft_margin * width_margin)
        features.append(draft_margin / max(ship_features.get('draft', 5), 0.1))  # 水深/吃水比
        
        return np.array(features).reshape(1, -1)
    
    def generate_training_data(self, edge_features_dict: Dict,
                               constraint_checker,
                               ship_templates: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成多任务训练数据
        
        Returns:
            X: 特征矩阵
            y_risk: 风险评分标签
            y_passable: 可达性标签（0/1）
        """
        from ship_navigator import ShipCharacteristics
        
        X_list = []
        y_risk_list = []
        y_passable_list = []
        
        for edge_key, edge_feat in edge_features_dict.items():
            for ship_template in ship_templates:
                features = self._extract_features(edge_feat, ship_template)
                X_list.append(features[0])
                
                # 创建 ShipCharacteristics 对象
                ship_obj = ShipCharacteristics(
                    ship_name="template",
                    draft=ship_template.get('draft', 5),
                    width=ship_template.get('width', 15),
                    height=ship_template.get('height', 20),
                    length=ship_template.get('length', 100),
                    tonnage=ship_template.get('tonnage', 5000),
                    max_speed=ship_template.get('max_speed', 15)
                )
                
                # 风险标签（规则计算）
                risk = constraint_checker._rule_based_risk_score(edge_key, ship_obj)
                y_risk_list.append(risk)
                
                # 可达性标签（规则判断）
                passable, _ = constraint_checker.check_edge_passable(edge_key, ship_obj)
                y_passable_list.append(1 if passable else 0)
        
        return np.array(X_list), np.array(y_risk_list), np.array(y_passable_list)
    
    def train(self, X: np.ndarray, y_risk: np.ndarray, y_passable: np.ndarray,
              epochs: int = 200, batch_size: int = 128, lr: float = 0.002,
              risk_weight: float = 0.5, passable_weight: float = 0.5,
              val_split: float = 0.2, patience: int = 20,
              label_smoothing: float = 0.05, noise_std: float = 0.0):
        """
        训练多任务模型（NLL改进版）
        
        基于文献：
        - DIR (NeurIPS 2021): 处理连续目标不平衡回归
        - UVOTE (GCPR 2024): NLL损失 + 方差预测
        
        风险回归用高斯NLL: loss = 0.5*(exp(-log_var)*(y-mu)^2 + log_var)
        可达性分类用BCEWithLogitsLoss + 标签平滑
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("训练多任务DNN模型（NLL改进版），样本数: %d", len(X))
        
        X_train, X_val, y_risk_train, y_risk_val, y_pass_train, y_pass_val = train_test_split(
            X, y_risk, y_passable, test_size=val_split, random_state=42, shuffle=True
        )
        
        logger.info("训练集: %d, 验证集: %d", len(X_train), len(X_val))
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_risk_train_tensor = torch.FloatTensor(y_risk_train).reshape(-1, 1)
        y_pass_train_tensor = torch.FloatTensor(y_pass_train).reshape(-1, 1)
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_risk_val_tensor = torch.FloatTensor(y_risk_val).reshape(-1, 1)
        y_pass_val_tensor = torch.FloatTensor(y_pass_val).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_risk_train_tensor, y_pass_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        passable_criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        best_val_r2 = -float('inf')
        best_epoch = 0
        no_improve_count = 0
        best_state = None
        
        try:
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_risk, batch_passable in train_loader:
                    optimizer.zero_grad()
                    
                    pred_risk_mu, pred_risk_log_var, pred_pass_logit = self.model(batch_X)
                    
                    risk_var = torch.exp(pred_risk_log_var).clamp(min=1e-4)
                    loss_risk = 0.5 * (pred_risk_log_var + (batch_risk - pred_risk_mu)**2 / risk_var)
                    loss_risk = loss_risk.mean()
                    
                    smoothed = batch_passable * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss_passable = passable_criterion(pred_pass_logit, smoothed)
                    
                    loss = risk_weight * loss_risk + passable_weight * loss_passable
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                self.model.eval()
                with torch.no_grad():
                    v_mu, v_lv, v_pl = self.model(X_val_tensor)
                    v_var = torch.exp(v_lv).clamp(min=1e-4)
                    vl_risk = 0.5 * (v_lv + (y_risk_val_tensor - v_mu)**2 / v_var).mean()
                    vl_pass = passable_criterion(v_pl, y_pass_val_tensor)
                    val_loss = risk_weight * vl_risk + passable_weight * vl_pass
                    
                    ss_res = torch.sum((y_risk_val_tensor - v_mu) ** 2)
                    ss_tot = torch.sum((y_risk_val_tensor - torch.mean(y_risk_val_tensor)) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    
                    v_pass_prob = torch.sigmoid(v_pl)
                    v_pred_bin = (v_pass_prob > 0.5).float()
                    accuracy = torch.mean((v_pred_bin == y_pass_val_tensor).float())
                    tp = torch.sum((v_pred_bin == 1) & (y_pass_val_tensor == 1)).float()
                    fp = torch.sum((v_pred_bin == 1) & (y_pass_val_tensor == 0)).float()
                    fn = torch.sum((v_pred_bin == 0) & (y_pass_val_tensor == 1)).float()
                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Epoch %d/%d | TrainLoss: %.4f | ValLoss: %.4f | "
                        "R²: %.4f | Acc: %.4f | F1: %.4f | LR: %.6f",
                        epoch + 1, epochs, train_loss / len(train_loader),
                        val_loss.item(), r2.item(), accuracy.item(), f1.item(),
                        optimizer.param_groups[0]['lr']
                    )
                
                improved = val_loss < best_val_loss or r2 > best_val_r2
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                if r2 > best_val_r2:
                    best_val_r2 = r2.item()
                
                if improved:
                    best_epoch = epoch
                    no_improve_count = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        logger.info("早停! 最佳epoch: %d, R²: %.4f", best_epoch + 1, best_val_r2)
                        break
            
            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.is_trained = True
            logger.info("多任务DNN训练完成, 最佳epoch: %d, R²: %.4f", best_epoch + 1, best_val_r2)
            self.save()
        except Exception as e:
            logger.error("DNN训练异常: %s", e)
            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.is_trained = True
                logger.info("从最佳检查点恢复, R²: %.4f", best_val_r2)
                self.save()
            raise
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = self.model_path.replace('.pt', f'_{name}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'is_trained': self.is_trained
        }, checkpoint_path)
    
    def _load_checkpoint(self, name: str):
        """加载检查点"""
        checkpoint_path = self.model_path.replace('.pt', f'_{name}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def predict(self, edge_features: Dict, ship_features: Dict) -> Tuple[float, float]:
        """
        预测风险评分和可达性概率
        
        Returns:
            (risk_score, passability_probability)
        """
        if not self.is_trained:
            logger.warning("多任务模型未训练，返回默认值")
            return 50.0, 0.5
        
        features = self._extract_features(edge_features, ship_features)
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)
        
        self.model.eval()
        with torch.no_grad():
            risk_mu, _, pass_logit = self.model(features_tensor)
        
        risk = float(np.clip(risk_mu[0][0], 0, 100))
        passable = float(torch.sigmoid(pass_logit[0][0]))
        return risk, passable
    
    def save(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'is_trained': self.is_trained
        }, self.model_path)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("多任务DNN模型已保存: %s", self.model_path)
    
    def load(self) -> bool:
        """加载模型"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.input_dim = checkpoint['input_dim']
                self.is_trained = checkpoint['is_trained']
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info("多任务DNN模型已加载: %s", self.model_path)
                return True
            except RuntimeError as e:
                logger.warning("模型架构不匹配，跳过加载（需重新训练）: %s", str(e)[:100])
                self.is_trained = False
                return False
        return False
