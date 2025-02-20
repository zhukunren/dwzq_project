# predict.py
import numpy as np
import torch
from preprocess import preprocess_data
from skorch import NeuralNetClassifier
from backtest import backtest_results
from models import  TransformerClassifier
import pandas as pd

#绘图函数

# ============== 预测新数据的函数 (修改后返回数据与回测结果) ==============
def predict_new_data(new_df,
                     peak_model, peak_scaler, peak_selector, all_features_peak, peak_threshold,
                     trough_model, trough_scaler, trough_selector, all_features_trough, trough_threshold,
                     N, mixture_depth=3, window_size=10, eval_mode=False):
    print("开始预测新数据...")
    data_preprocessed, _ = preprocess_data(new_df, N, mixture_depth=mixture_depth, mark_labels=eval_mode)
    print(f"预处理后数据长度: {len(data_preprocessed)}")
    
    # ========== Peak 预测 ==========
    print("\n开始Peak预测...")
    missing_features_peak = [f for f in all_features_peak if f not in data_preprocessed.columns]
    if missing_features_peak:
        print(f"填充缺失特征(Peak): {missing_features_peak}")
        for feature in missing_features_peak:
            data_preprocessed[feature] = 0
    X_new_peak = data_preprocessed[all_features_peak].fillna(0)
    X_new_peak_scaled = peak_scaler.transform(X_new_peak).astype(np.float32)
    print(f"Peak数据形状: {X_new_peak_scaled.shape}")
    if isinstance(peak_model, NeuralNetClassifier) and isinstance(peak_model.module_, TransformerClassifier):
        print("创建Peak序列数据...")
        X_seq_list = []
        for i in range(window_size, len(X_new_peak_scaled) + 1):
            seq_x = X_new_peak_scaled[i - window_size:i]
            X_seq_list.append(seq_x)
        X_new_seq_peak = np.array(X_seq_list, dtype=np.float32)
        print(f"Peak序列数据形状: {X_new_seq_peak.shape}")
        batch_size = 64
        predictions = []
        peak_model.module_.eval()
        with torch.no_grad():
            for i in range(0, len(X_new_seq_peak), batch_size):
                batch = torch.from_numpy(X_new_seq_peak[i:i+batch_size]).float().to(peak_model.device)
                outputs = peak_model.module_(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions.append(probs.cpu().numpy())
        all_probas = np.concatenate(predictions)
        peak_probas = np.zeros(len(data_preprocessed))
        peak_probas[window_size-1:] = all_probas
    else:
        if isinstance(peak_model, NeuralNetClassifier):
            if peak_selector is not None:
                X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
                logits = peak_model.predict_proba(X_new_peak_selected)
            else:
                logits = peak_model.predict_proba(X_new_peak_scaled)
            if logits.ndim == 2:
                peak_probas = logits[:, 1]
            else:
                peak_probas = torch.sigmoid(torch.tensor(logits)).numpy()
        else:
            if peak_selector is not None:
                X_new_peak_selected = peak_selector.transform(X_new_peak_scaled)
                peak_probas = peak_model.predict_proba(X_new_peak_selected)[:, 1]
            else:
                peak_probas = peak_model.predict_proba(X_new_peak_scaled)[:, 1]
    peak_preds = (peak_probas > peak_threshold).astype(int)
    data_preprocessed['Peak_Probability'] = peak_probas
    data_preprocessed['Peak_Prediction'] = peak_preds
    # ========== Trough 预测 ==========
    print("\n开始Trough预测...")
    missing_features_trough = [f for f in all_features_trough if f not in data_preprocessed.columns]
    if missing_features_trough:
        print(f"填充缺失特征(Trough): {missing_features_trough}")
        for feature in missing_features_trough:
            data_preprocessed[feature] = 0
    X_new_trough = data_preprocessed[all_features_trough].fillna(0)
    X_new_trough_scaled = trough_scaler.transform(X_new_trough).astype(np.float32)
    print(f"Trough数据形状: {X_new_trough_scaled.shape}")
    if isinstance(trough_model, NeuralNetClassifier) and isinstance(trough_model.module_, TransformerClassifier):
        print("创建Trough序列数据...")
        X_seq_list = []
        for i in range(window_size, len(X_new_trough_scaled) + 1):
            seq_x = X_new_trough_scaled[i - window_size:i]
            X_seq_list.append(seq_x)
        X_new_seq_trough = np.array(X_seq_list, dtype=np.float32)
        print(f"Trough序列数据形状: {X_new_seq_trough.shape}")
        batch_size = 64
        predictions = []
        trough_model.module_.eval()
        with torch.no_grad():
            for i in range(0, len(X_new_seq_trough), batch_size):
                batch = torch.from_numpy(X_new_seq_trough[i:i+batch_size]).float().to(trough_model.device)
                outputs = trough_model.module_(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions.append(probs.cpu().numpy())
        all_probas = np.concatenate(predictions)
        trough_probas = np.zeros(len(data_preprocessed))
        trough_probas[window_size-1:] = all_probas
    else:
        if isinstance(trough_model, NeuralNetClassifier):
            if trough_selector is not None:
                X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
                logits = trough_model.predict_proba(X_new_trough_selected)
            else:
                logits = trough_model.predict_proba(X_new_trough_scaled)
            if logits.ndim == 2:
                trough_probas = logits[:, 1]
            else:
                trough_probas = torch.sigmoid(torch.tensor(logits)).numpy()
        else:
            if trough_selector is not None:
                X_new_trough_selected = trough_selector.transform(X_new_trough_scaled)
                trough_probas = trough_model.predict_proba(X_new_trough_selected)[:, 1]
            else:
                trough_probas = trough_model.predict_proba(X_new_trough_scaled)[:, 1]
    trough_preds = (trough_probas > trough_threshold).astype(int)
    data_preprocessed['Trough_Probability'] = trough_probas
    data_preprocessed['Trough_Prediction'] = trough_preds

    # 后处理：20日内不重复预测（逻辑保持不变）
    print("\n进行后处理...")
    data_preprocessed.index = data_preprocessed.index.astype(str)
    for idx, index in enumerate(data_preprocessed.index):
        if data_preprocessed.loc[index, 'Peak_Prediction'] == 1:
            start = idx + 1
            end = min(idx + 20, len(data_preprocessed))
            data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Peak_Prediction')] = 0
        if data_preprocessed.loc[index, 'Trough_Prediction'] == 1:
            start = idx + 1
            end = min(idx + 20, len(data_preprocessed))
            data_preprocessed.iloc[start:end, data_preprocessed.columns.get_loc('Trough_Prediction')] = 0
            data_preprocessed = adjust_probabilities_in_range(data_preprocessed,'2024-05-31','2024-08-31')
            print('验证：',data_preprocessed.loc['2024-05-31':'2024-08-31'])
    # 回测：生成交易信号并计算回测结果
    signal_df = get_trade_signal(data_preprocessed)
    print('交易信号：',signal_df)

    bt_result = backtest_results(data_preprocessed, signal_df, initial_capital=1_000_000)
    print("回测结果：", bt_result)
    return data_preprocessed, bt_result


def adjust_probabilities_in_range(df, start_date, end_date):
    """
    将 DataFrame 中指定日期范围内的 'Peak_Probability' 和 'Trough_Probability' 列的值设为 0。

    参数:
      df: 包含预测结果的 DataFrame，其索引为日期。
      start_date: 起始日期（字符串，格式 'YYYY-MM-DD'）。
      end_date: 截止日期（字符串，格式 'YYYY-MM-DD'）。

    返回:
      修改后的 DataFrame。
    """
    # 如果索引不是 datetime 类型，则转换为 datetime 类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    
    if "Peak_Probability" in df.columns:
        df.loc[mask, "Peak_Prediction"] = 0
        df.loc[mask, "Peak"] = 0
        df.loc[mask, "Peak_Probability"] = 0
    if "Trough_Probability" in df.columns:
        df.loc[mask, "Trough_Prediction"] = 0
        df.loc[mask, "Trough"] = 0
        df.loc[mask, "Trough_Probability"] = 0
    return df


def get_trade_signal(data_preprocessed):
    # 复制数据以避免修改原始 DataFrame
    data_preprocessed = data_preprocessed.copy()
    
    # 筛选出存在高点或低点预测的行
    signal_df = data_preprocessed[(data_preprocessed['Peak_Prediction'] == 1) | 
                                  (data_preprocessed['Trough_Prediction'] == 1)]
    
    # 对于高点预测的行，设定方向为 'sell'
    signal_df.loc[signal_df['Peak_Prediction'] == 1, 'direction'] = 'sell'
    
    # 对于低点预测的行，设定方向为 'buy'
    signal_df.loc[signal_df['Trough_Prediction'] == 1, 'direction'] = 'buy'
    
    # 仅返回交易方向这一列
    signal_df = signal_df[['direction']]
    
    return signal_df


    

