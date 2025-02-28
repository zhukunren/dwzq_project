# train.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, average_precision_score, matthews_corrcoef, roc_auc_score, f1_score, classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from joblib import Parallel, delayed, parallel_backend

from preprocess import create_pos_neg_sequences_by_consecutive_labels
from models import get_transformer_classifier, get_mlp_classifier
#import streamlit as st

def identity_transform(x):
    return x

#@st.cache_data
def optimize_threshold(y_true, y_proba, metric='precision'):
    """
    根据给定评估指标（'precision', 'f1', 'recall', 'accuracy', 'mcc'）在 [0,1] 区间内寻找最佳分类阈值。
    
    参数：
        y_true (array-like): 真实标签（0 或 1）。
        y_proba (array-like): 预测的正类概率。
        metric (str): 评估指标，支持 'precision', 'f1', 'recall', 'accuracy', 'mcc'。
        
    返回：
        best_thresh (float): 使指定指标达到最佳的分类阈值。
    """
    best_thresh = 0.5
    best_score = -1
    for thresh in np.linspace(0, 1, 101):
        y_pred_temp = (y_proba > thresh).astype(int)
        if metric == 'precision':
            score = precision_score(y_true, y_pred_temp)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred_temp)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_temp)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred_temp)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred_temp)
        else:
            raise ValueError("metric must be one of 'precision', 'f1', 'recall', 'accuracy', 'mcc'")
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_thresh

def train_model_for_label(df, N, label_column, all_features, classifier_name, n_features_selected, window_size=30, oversample_method='SMOTE', class_weight=None):
    print(f"开始训练 {label_column} 模型...")
    data = df.copy()
    X = data[all_features]
    y = data[label_column].astype(np.int64)
    # 特征相关性过滤
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"检测到高相关特征 {len(to_drop)} 个，将进行剔除。")
    else:
        print("未检测到高相关特征。")
    all_features_filtered = [f for f in all_features if f not in to_drop]
    X = data[all_features_filtered]
    print(f"过滤后特征数量：{len(all_features_filtered)}")
    print("标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    if classifier_name == 'Transformer':
        print("使用时间顺序强化采样构造时序数据...")
        X_features, y_features = create_pos_neg_sequences_by_consecutive_labels(X_scaled, y, negative_ratio=1.0, adjacent_steps=5)
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_features, test_size=0.2, random_state=42, stratify=y_features)
        #print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    else:  # 对于 MLP 模型
        if oversample_method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif oversample_method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        elif oversample_method == 'Borderline-SMOTE':
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif oversample_method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=42)
        elif oversample_method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=42)
        elif oversample_method in ['Class Weights', 'None']:
            sampler = None
        else:
            raise ValueError(f"未知的过采样方法: {oversample_method}")
        if sampler is not None:
            X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)
            #print(f"数据形状: X={X_resampled.shape}, y={y_resampled.shape}")
        else:
            print("不进行过采样，使用原始数据。")
            X_resampled, y_resampled = X_scaled, y
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
        #print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    num_features = X_train.shape[-1]
    if classifier_name == 'Transformer':
        clf_name, clf = get_transformer_classifier(num_features=num_features, window_size=window_size, class_weights=None)
    elif classifier_name == 'MLP':
        clf_name, clf = get_mlp_classifier(input_dim=num_features, class_weights=None)
    else:
        raise ValueError("仅支持 Transformer 和 MLP 模型")
    
    print(f"正在为分类器 {clf_name} 进行 GridSearchCV 调参...")
    param_grid = {}
    if clf_name == 'transformer':
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [10],
            'module__hidden_dim': [128]
        }
    elif clf_name == 'mlp':
        param_grid = {
            'lr': [1e-3],
            'max_epochs': [30]
        }
    
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='precision' if classifier_name == 'transformer' else "f1" ,
        verbose=0,
        error_score='raise'
    )
    try:
        grid_search.fit(X_train, y_train)
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳得分: {grid_search.best_score_:.4f}")
        best_estimator = grid_search.best_estimator_
    except Exception as e:
        print(f"GridSearchCV 失败: {e}")
        raise
    
    print("不进行特征选择，使用全部特征")
    feature_selector = FunctionTransformer(func=identity_transform, validate=False)
    selected_features = all_features_filtered.copy()
    
    if isinstance(best_estimator, type(clf)):
        if feature_selector is not None:
            X_test_selected = feature_selector.transform(X_test)
            logits = best_estimator.predict_proba(X_test_selected)
        else:
            logits = best_estimator.predict_proba(X_test)
        if isinstance(logits, np.ndarray) and logits.ndim == 2:
            y_proba = logits[:, 1]
        else:
            y_proba = F.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    else:
        if feature_selector is not None:
            X_test_selected = feature_selector.transform(X_test)
            y_proba = best_estimator.predict_proba(X_test_selected)[:, 1]
        else:
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
    
    best_thresh = optimize_threshold(y_test, y_proba,metric='accuracy' if  classifier_name == 'MLP' else 'precision' )#可指定（'precision', 'f1', 'recall', 'accuracy', 'mcc'）
    y_pred = (y_proba > best_thresh).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    metrics = {
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }
    return (best_estimator, scaler, feature_selector, selected_features, all_features_filtered, grid_search.best_score_, metrics, best_thresh)

#@st.cache_resource
def train_model(df_preprocessed, N, all_features, classifier_name, mixture_depth, n_features_selected, oversample_method, window_size=30):
    print("开始训练模型...")
    data = df_preprocessed.copy()
    #print(f"预处理后数据长度: {len(data)}")
    labels = ['Peak', 'Trough']
    with parallel_backend('threading', n_jobs=-1):
        results = Parallel()(
            delayed(train_model_for_label)(
                data, N, label, all_features, classifier_name, n_features_selected, window_size, oversample_method, class_weight='balanced' if oversample_method == 'Class Weights' else None
            )
            for label in labels
        )
    peak_results = results[0]
    trough_results = results[1]
    (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score, peak_metrics, peak_threshold) = peak_results
    (trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_best_score, trough_metrics, trough_threshold) = trough_results
    return (peak_model, peak_scaler, peak_selector, peak_selected_features, all_features_peak, peak_best_score, peak_metrics, peak_threshold,
            trough_model, trough_scaler, trough_selector, trough_selected_features, all_features_trough, trough_best_score, trough_metrics, trough_threshold)
