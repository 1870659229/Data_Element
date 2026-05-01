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
except ImportError:
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
    多任务深度神经网络（超强版）
    
    架构改进：
    - 更深的网络（6层共享层）
    - 更宽的层（512->256->128->64->32）
    - Swish激活函数（比ReLU更平滑）
    - 残差连接（ResNet风格）
    - 注意力机制（多头注意力）
    - LayerNorm + Dropout组合
    - 多尺度特征融合
    
    同时预测：
    - 风险评分（回归任务，0-100）
    - 可达性概率（分类任务，0-1）
    """
    
    def __init__(self, input_dim=20, shared_dims=[512, 256, 128, 64, 32], dropout_rate=0.5):
        super(MultiTaskDNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.shared_dims = shared_dims
        
        # 输入投影层（将输入映射到高维空间）
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, shared_dims[0]),
            nn.LayerNorm(shared_dims[0]),
            nn.SiLU(),  # Swish激活
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 构建共享层（带残差连接和投影）
        self.shared_layers = nn.ModuleList()
        self.shared_norms = nn.ModuleList()
        self.shared_dropouts = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        prev_dim = shared_dims[0]
        for dim in shared_dims[1:]:
            self.shared_layers.append(nn.Linear(prev_dim, dim))
            self.shared_norms.append(nn.LayerNorm(dim))
            self.shared_dropouts.append(nn.Dropout(dropout_rate))
            # 残差投影（处理维度不匹配）
            if prev_dim != dim:
                self.residual_projections.append(nn.Linear(prev_dim, dim))
            else:
                self.residual_projections.append(None)
            prev_dim = dim
        
        # 多尺度特征融合
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(sum(shared_dims[1:]), prev_dim),
            nn.LayerNorm(prev_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 多头注意力机制（学习特征重要性）
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(prev_dim // 2),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.SiLU(),
            nn.Linear(prev_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 风险预测头（更深更宽）
        self.risk_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1，再乘以100
        )
        
        # 可达性预测头（更深更宽）
        self.passability_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1概率
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """He初始化 + 特殊层处理"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 收集多尺度特征
        multi_scale_features = []
        
        # 共享层（带残差）
        for layer, norm, dropout, res_proj in zip(
            self.shared_layers, self.shared_norms, 
            self.shared_dropouts, self.residual_projections
        ):
            residual = x
            x = layer(x)
            x = norm(x)
            x = F.silu(x)  # Swish激活
            x = dropout(x)
            # 残差连接
            if res_proj is not None:
                residual = res_proj(residual)
            x = x + residual
            multi_scale_features.append(x)
        
        # 多尺度特征融合
        fused_features = torch.cat(multi_scale_features, dim=-1)
        x = self.multi_scale_fusion(fused_features)
        
        # 注意力权重
        attn_weight = self.attention(x)
        shared_features = x * attn_weight
        
        # 任务特定输出
        risk = self.risk_head(shared_features) * 100
        passable = self.passability_head(shared_features)
        return risk, passable


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
              epochs: int = 300, batch_size: int = 64, lr: float = 0.001,
              risk_weight: float = 0.6, passable_weight: float = 0.4,
              val_split: float = 0.2, patience: int = 30,
              label_smoothing: float = 0.05, noise_std: float = 0.01):
        """
        训练多任务模型（超强版超参数）
        
        Args:
            X: 特征矩阵
            y_risk: 风险评分标签（0-100）
            y_passable: 可达性标签（0/1）
            epochs: 最大训练轮数（增加到300）
            batch_size: 批次大小（减小到64，更稳定）
            lr: 学习率
            risk_weight: 风险任务损失权重
            passable_weight: 可达性任务损失权重
            val_split: 验证集比例
            patience: 早停耐心值（增加到30）
            label_smoothing: 标签平滑系数
            noise_std: 数据增强噪声标准差
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("训练多任务DNN模型（超强版），样本数: %d", len(X))
        
        # 划分训练集和验证集
        X_train, X_val, y_risk_train, y_risk_val, y_pass_train, y_pass_val = train_test_split(
            X, y_risk, y_passable, test_size=val_split, random_state=42, shuffle=True
        )
        
        logger.info("训练集: %d, 验证集: %d", len(X_train), len(X_val))
        
        # 标准化（只用训练集拟合）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_risk_train_tensor = torch.FloatTensor(y_risk_train).reshape(-1, 1)
        y_pass_train_tensor = torch.FloatTensor(y_pass_train).reshape(-1, 1)
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_risk_val_tensor = torch.FloatTensor(y_risk_val).reshape(-1, 1)
        y_pass_val_tensor = torch.FloatTensor(y_pass_val).reshape(-1, 1)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_risk_train_tensor, y_pass_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # 优化器：AdamW（比Adam更好的权重衰减）
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-3,  # 更强的权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度：预热 + 余弦退火 + 周期性重启
        warmup_epochs = 15
        scheduler_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-7
        )
        
        # 损失函数
        # 风险预测：SmoothL1Loss（对异常值更鲁棒）
        risk_criterion = nn.SmoothL1Loss(beta=5.0)
        # 可达性预测：BCELoss + 标签平滑
        passable_criterion = nn.BCELoss()
        
        # 早停相关变量
        best_val_loss = float('inf')
        best_val_risk_loss = float('inf')
        best_val_passable_loss = float('inf')
        best_val_r2 = -float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        no_improve_count = 0
        
        # 训练循环
        for epoch in range(epochs):
            # ===== 训练阶段 =====
            self.model.train()
            train_loss = 0
            train_risk_loss = 0
            train_passable_loss = 0
            
            for batch_X, batch_risk, batch_passable in train_loader:
                # 数据增强：添加高斯噪声
                if noise_std > 0:
                    noise = torch.randn_like(batch_X) * noise_std
                    batch_X = batch_X + noise
                
                optimizer.zero_grad()
                
                # 前向传播
                pred_risk, pred_passable = self.model(batch_X)
                
                # 标签平滑（仅对分类任务）
                smoothed_passable = batch_passable * (1 - label_smoothing) + 0.5 * label_smoothing
                
                # 计算损失
                loss_risk = risk_criterion(pred_risk, batch_risk)
                loss_passable = passable_criterion(pred_passable, smoothed_passable)
                
                # 加权总损失
                loss = risk_weight * loss_risk + passable_weight * loss_passable
                
                # 反向传播
                loss.backward()
                # 梯度裁剪（更强的裁剪）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                train_risk_loss += loss_risk.item()
                train_passable_loss += loss_passable.item()
            
            # ===== 验证阶段 =====
            self.model.eval()
            with torch.no_grad():
                val_pred_risk, val_pred_passable = self.model(X_val_tensor)
                val_loss_risk = risk_criterion(val_pred_risk, y_risk_val_tensor)
                val_loss_passable = passable_criterion(val_pred_passable, y_pass_val_tensor)
                val_loss = risk_weight * val_loss_risk + passable_weight * val_loss_passable
                
                # 计算验证集上的R²（风险预测）
                ss_res = torch.sum((y_risk_val_tensor - val_pred_risk) ** 2)
                ss_tot = torch.sum((y_risk_val_tensor - torch.mean(y_risk_val_tensor)) ** 2)
                r2 = 1 - ss_res / ss_tot
                
                # 计算验证集上的准确率（可达性预测）
                val_pred_passable_binary = (val_pred_passable > 0.5).float()
                accuracy = torch.mean((val_pred_passable_binary == y_pass_val_tensor).float())
                
                # 计算F1分数
                tp = torch.sum((val_pred_passable_binary == 1) & (y_pass_val_tensor == 1)).float()
                fp = torch.sum((val_pred_passable_binary == 1) & (y_pass_val_tensor == 0)).float()
                fn = torch.sum((val_pred_passable_binary == 0) & (y_pass_val_tensor == 1)).float()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # 学习率调整
            if epoch < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()
            
            # 记录日志
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f (Risk: %.4f, Pass: %.4f) | "
                    "Val Loss: %.4f (Risk: %.4f, Pass: %.4f) | "
                    "Val R²: %.4f | Val Acc: %.4f | Val F1: %.4f | LR: %.6f",
                    epoch + 1, epochs,
                    train_loss / len(train_loader),
                    train_risk_loss / len(train_loader),
                    train_passable_loss / len(train_loader),
                    val_loss.item(),
                    val_loss_risk.item(),
                    val_loss_passable.item(),
                    r2.item(),
                    accuracy.item(),
                    f1.item(),
                    optimizer.param_groups[0]['lr']
                )
            
            # 早停检查（基于综合验证损失 + R²）
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_risk_loss = val_loss_risk.item()
                best_val_passable_loss = val_loss_passable.item()
                improved = True
            
            if r2 > best_val_r2:
                best_val_r2 = r2.item()
                improved = True
            
            if accuracy > best_val_acc:
                best_val_acc = accuracy.item()
                improved = True
            
            if improved:
                best_epoch = epoch
                no_improve_count = 0
                # 保存最佳模型
                self._save_checkpoint('best')
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(
                        "早停触发！最佳轮数: %d, 最佳Val Loss: %.4f (Risk: %.4f, Pass: %.4f) | "
                        "最佳R²: %.4f | 最佳Acc: %.4f",
                        best_epoch + 1, best_val_loss, best_val_risk_loss, best_val_passable_loss,
                        best_val_r2, best_val_acc
                    )
                    break
        
        # 加载最佳模型
        self._load_checkpoint('best')
        self.is_trained = True
        logger.info("多任务DNN模型训练完成，最佳轮数: %d", best_epoch + 1)
        
        # 保存最终模型
        self.save()
    
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
            risk, passable = self.model(features_tensor)
        
        return float(risk[0][0]), float(passable[0][0])
    
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
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.input_dim = checkpoint['input_dim']
            self.is_trained = checkpoint['is_trained']
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info("多任务DNN模型已加载: %s", self.model_path)
            return True
        return False
