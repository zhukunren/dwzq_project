# preprocess.py
import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from math import isnan
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score,f1_score

# 从外部 function.py 导入技术指标计算函数
# 请确保你的 function.py 文件中包含 compute_RSI, compute_MACD, compute_KD, compute_momentum, compute_ROC, compute_Bollinger_Bands,
# compute_ATR, compute_volatility, compute_OBV, compute_VWAP, compute_MFI, compute_CMF, compute_chaikin_oscillator,
# compute_CCI, compute_williams_r, compute_zscore, compute_ADX, compute_TRIX, compute_ultimate_oscillator, compute_PPO,
# compute_DPO, compute_KST, compute_KAMA, compute_EMA, compute_MoneyFlowIndex, identify_low_troughs, identify_high_peaks,
# compute_SMA, compute_PercentageB, compute_AccumulationDistribution, compute_HighLow_Spread, compute_PriceChannel, compute_RenkoSlope
from function import *
import streamlit as st


def preprocess_data(data, N, mixture_depth, mark_labels=True, min_features_to_select=10, max_features_for_mixture=50):
    print("开始预处理数据...")
    print(data.head())
    data = data.sort_values('TradeDate').copy()
    data.index = pd.to_datetime(data['TradeDate'], format='%Y%m%d')
    
    # 基础特征计算
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Price_MA20_Diff'] = (data['Close'] - data['MA_20']) / data['MA_20']
    data['MA5_MA20_Cross'] = np.where(data['MA_5'] > data['MA_20'], 1, 0)
    data['MA5_MA20_Cross_Diff'] = data['MA5_MA20_Cross'].diff()
    data['Slope_MA5'] = data['MA_5'].diff()
    data['RSI_14'] = compute_RSI(data['Close'], period=14)
    data['MACD'], data['MACD_signal'] = compute_MACD(data['Close'])
    data['MACD_Cross'] = np.where(data['MACD'] > data['MACD_signal'], 1, 0)
    data['MACD_Cross_Diff'] = data['MACD_Cross'].diff()
    data['K'], data['D'] = compute_KD(data['High'], data['Low'], data['Close'], period=14)
    data['Momentum_10'] = compute_momentum(data['Close'], period=10)
    data['ROC_10'] = compute_ROC(data['Close'], period=10)
    data['RSI_Reversal'] = (data['RSI_14'] > 70).astype(int) - (data['RSI_14'] < 30).astype(int)
    data['Reversal_Signal'] = (data['Close'] > data['High'].rolling(window=10).max()).astype(int) - (data['Close'] < data['Low'].rolling(window=10).min()).astype(int)
    data['UpperBand'], data['MiddleBand'], data['LowerBand'] = compute_Bollinger_Bands(data['Close'], period=20)
    data['ATR_14'] = compute_ATR(data['High'], data['Low'], data['Close'], period=14)
    data['Volatility_10'] = compute_volatility(data['Close'], period=10)
    data['Bollinger_Width'] = (data['UpperBand'] - data['LowerBand']) / data['MiddleBand']
    
    if 'Volume' in data.columns:
        data['OBV'] = compute_OBV(data['Close'], data['Volume'])
        data['Volume_Change'] = data['Volume'].pct_change()
        data['VWAP'] = compute_VWAP(data['High'], data['Low'], data['Close'], data['Volume'])
        data['MFI_14'] = compute_MFI(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
        data['CMF_20'] = compute_CMF(data['High'], data['Low'], data['Close'], data['Volume'], period=20)
        data['Chaikin_Osc'] = compute_chaikin_oscillator(data['High'], data['Low'], data['Close'], data['Volume'], short_period=3, long_period=10)
    else:
        data['OBV'] = np.nan
        data['Volume_Change'] = np.nan
        data['VWAP'] = np.nan
        data['MFI_14'] = np.nan
        data['CMF_20'] = np.nan
        data['Chaikin_Osc'] = np.nan
        
    data['CCI_20'] = compute_CCI(data['High'], data['Low'], data['Close'], period=20)
    data['Williams_%R_14'] = compute_williams_r(data['High'], data['Low'], data['Close'], period=14)
    data['ZScore_20'] = compute_zscore(data['Close'], period=20)
    data['Price_Mean_Diff'] = (data['Close'] - data['Close'].rolling(window=10).mean()) / data['Close'].rolling(window=10).mean()
    data['High_Mean_Diff'] = (data['High'] - data['High'].rolling(window=10).mean()) / data['High'].rolling(window=10).mean()
    data['Low_Mean_Diff'] = (data['Low'] - data['Low'].rolling(window=10).mean()) / data['Low'].rolling(window=10).mean()
    data['Plus_DI'], data['Minus_DI'], data['ADX_14'] = compute_ADX(data['High'], data['Low'], data['Close'], period=14)
    data['TRIX_15'] = compute_TRIX(data['Close'], period=15)
    data['Ultimate_Osc'] = compute_ultimate_oscillator(data['High'], data['Low'], data['Close'], short_period=7, medium_period=14, long_period=28)
    data['PPO'] = compute_PPO(data['Close'], fast_period=12, slow_period=26)
    data['DPO_20'] = compute_DPO(data['Close'], period=20)
    data['KST'], data['KST_signal'] = compute_KST(data['Close'], r1=10, r2=15, r3=20, r4=30, sma1=10, sma2=10, sma3=10, sma4=15)
    data['KAMA_10'] = compute_KAMA(data['Close'], n=10, pow1=2, pow2=30)
    data['Seasonality'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['one'] = 1

    # 新增更多样化特征
    data['SMA_10'] = compute_SMA(data['Close'], window=10)
    data['SMA_30'] = compute_SMA(data['Close'], window=30)
    data['EMA_10'] = compute_EMA(data['Close'], span=10)
    data['EMA_30'] = compute_EMA(data['Close'], span=30)
    data['PercentB'] = compute_PercentageB(data['Close'], data['UpperBand'], data['LowerBand'])
    if 'Volume' in data.columns:
        data['AccumDist'] = compute_AccumulationDistribution(data['High'], data['Low'], data['Close'], data['Volume'])
    else:
        data['AccumDist'] = np.nan
    if 'Volume' in data.columns:
        data['MFI_New'] = compute_MoneyFlowIndex(data['High'], data['Low'], data['Close'], data['Volume'], period=14)
    else:
        data['MFI_New'] = np.nan
    data['HL_Spread'] = compute_HighLow_Spread(data['High'], data['Low'])
    price_channel = compute_PriceChannel(data['High'], data['Low'], data['Close'], window=20)
    data['PriceChannel_Mid'] = price_channel['middle_channel']
    data['RenkoSlope'] = compute_RenkoSlope(data['Close'], bricks=3)
    
    # 标签生成
    if mark_labels:
        print("寻找局部高点和低点(仅训练阶段)...")
        N = int(N)
        data = identify_low_troughs(data, N)
        data = identify_high_peaks(data, N)
    else:
        if 'Peak' in data.columns:
            data.drop(columns=['Peak'], inplace=True)
        if 'Trough' in data.columns:
            data.drop(columns=['Trough'], inplace=True)
        data['Peak'] = 0
        data['Trough'] = 0
        
    print("添加计数指标...")
    data['PriceChange'] = data['Close'].diff()
    data['Up'] = np.where(data['PriceChange'] > 0, 1, 0)
    data['Down'] = np.where(data['PriceChange'] < 0, 1, 0)
    data['ConsecutiveUp'] = data['Up'] * (data['Up'].groupby((data['Up'] != data['Up'].shift()).cumsum()).cumcount() + 1)
    data['ConsecutiveDown'] = data['Down'] * (data['Down'].groupby((data['Down'] != data['Down'].shift()).cumsum()).cumcount() + 1)
    window_size = 10
    data['Cross_MA5'] = np.where(data['Close'] > data['MA_5'], 1, 0)
    data['Cross_MA5_Count'] = data['Cross_MA5'].rolling(window=window_size).sum()
    if 'Volume' in data.columns:
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_Spike'] = np.where(data['Volume'] > data['Volume_MA_5'] * 1.5, 1, 0)
        data['Volume_Spike_Count'] = data['Volume_Spike'].rolling(window=10).sum()
    else:
        data['Volume_Spike_Count'] = np.nan
    print("构建基础因子...")
    data['Close_MA5_Diff'] = data['Close'] - data['MA_5']
    data['Pch'] = data['Close'] / data['Close'].shift(1) - 1
    data['MA5_MA20_Diff'] = data['MA_5'] - data['MA_20']
    data['RSI_Signal'] = data['RSI_14'] - 50
    data['MACD_Diff'] = data['MACD'] - data['MACD_signal']
    band_range = (data['UpperBand'] - data['LowerBand']).replace(0, np.nan)
    data['Bollinger_Position'] = (data['Close'] - data['MiddleBand']) / band_range
    data['Bollinger_Position'] = data['Bollinger_Position'].fillna(0)
    data['K_D_Diff'] = data['K'] - data['D']
    base_features = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff', 'ConsecutiveUp', 'ConsecutiveDown',
        'Cross_MA5_Count', 'Volume_Spike_Count','one','Close','Pch','CCI_20'
    ]
    base_features.extend([
        'Williams_%R_14', 'OBV', 'VWAP','ZScore_20', 'Plus_DI', 'Minus_DI',
        'ADX_14','Bollinger_Width', 'Slope_MA5', 'Volume_Change', 
        'Price_Mean_Diff','High_Mean_Diff','Low_Mean_Diff','MA_5','MA_20','MA_50',
        'MA_200','EMA_5','EMA_20'
    ])
    base_features.extend([
        'MFI_14','CMF_20','TRIX_15','Ultimate_Osc','Chaikin_Osc','PPO',
        'DPO_20','KST','KST_signal','KAMA_10'
    ])
    if 'Volume' in data.columns:
        base_features.append('Volume')
    print("对基础特征进行方差过滤...")
    X_base = data[base_features].fillna(0)
    selector = VarianceThreshold(threshold=0.0001)
    selector.fit(X_base)
    filtered_features = [f for f, s in zip(base_features, selector.get_support()) if s]
    print(f"方差过滤后剩余特征数：{len(filtered_features)}（从{len(base_features)}减少）")
    base_features = filtered_features
    print("对基础特征进行相关性过滤...")
    corr_matrix = data[base_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    base_features = [f for f in base_features if f not in to_drop]
    print(f"相关性过滤后剩余特征数：{len(base_features)}")
    print(f"生成混合因子，混合深度为 {mixture_depth}...")
    if mixture_depth > 1:
        operators = ['+', '-', '*', '/']
        mixed_features = base_features.copy()
        current_depth_features = base_features.copy()
        for depth in range(2, mixture_depth + 1):
            print(f"生成深度 {depth} 的混合因子...")
            new_features = []
            feature_pairs = combinations(current_depth_features, 2)
            for f1, f2 in feature_pairs:
                for op in operators:
                    new_feature_name = f'({f1}){op}({f2})_d{depth}'
                    try:
                        if op == '+':
                            data[new_feature_name] = data[f1] + data[f2]
                        elif op == '-':
                            data[new_feature_name] = data[f1] - data[f2]
                        elif op == '*':
                            data[new_feature_name] = data[f1] * data[f2]
                        elif op == '/':
                            denom = data[f2].replace(0, np.nan)
                            data[new_feature_name] = data[f1] / denom
                        data[new_feature_name] = data[new_feature_name].replace([np.inf, -np.inf], np.nan).fillna(0)
                        new_features.append(new_feature_name)
                    except Exception as e:
                        print(f"无法计算特征 {new_feature_name}，错误：{e}")
            if new_features:
                X_new = data[new_features].fillna(0)
                selector = VarianceThreshold(threshold=0.0001)
                selector.fit(X_new)
                new_features = [nf for nf, s in zip(new_features, selector.get_support()) if s]
                if len(new_features) > 1:
                    corr_matrix_new = data[new_features].corr().abs()
                    upper_new = corr_matrix_new.where(np.triu(np.ones(corr_matrix_new.shape), k=1).astype(bool))
                    to_drop_new = [column for column in upper_new.columns if any(upper_new[column] > 0.95)]
                    new_features = [f for f in new_features if f not in to_drop_new]
            mixed_features.extend(new_features)
            current_depth_features = new_features.copy()
        all_features = mixed_features.copy()
        print("进行 PCA 降维...")
        pca_components = min(100, len(all_features))
        pca = PCA(n_components=pca_components)
        X_mixed = data[all_features].fillna(0).values
        X_mixed_pca = pca.fit_transform(X_mixed)
        pca_feature_names = [f'PCA_{i}' for i in range(pca_components)]
        for i in range(pca_components):
            data[pca_feature_names[i]] = X_mixed_pca[:, i]
        all_features = pca_feature_names
    else:
        all_features = base_features.copy()
    print(f"最终特征数量：{len(all_features)}")
    required_cols = [
        'Close_MA5_Diff', 'MA5_MA20_Diff', 'RSI_Signal', 'MACD_Diff',
        'Bollinger_Position', 'K_D_Diff'
    ]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"列 {col} 未被创建，请检查数据和计算步骤。")
    print("删除缺失值...")
    initial_length = len(data)
    data = data.dropna().copy()
    final_length = len(data)
    print(f"数据预处理前长度: {initial_length}, 数据预处理后长度: {final_length}")
    return data, all_features

@st.cache_data
def create_pos_neg_sequences_by_consecutive_labels(X, y, negative_ratio=1.0, adjacent_steps=5):
    pos_idx = np.where(y == 1)[0]
    pos_segments = []
    if len(pos_idx) > 0:
        start = pos_idx[0]
        for i in range(1, len(pos_idx)):
            if pos_idx[i] != pos_idx[i-1] + 1:
                pos_segments.append(np.arange(start, pos_idx[i-1]+1))
                start = pos_idx[i]
        pos_segments.append(np.arange(start, pos_idx[-1]+1))
    pos_features = np.array([X[seg].mean(axis=0) for seg in pos_segments])
    pos_labels = np.ones(len(pos_features), dtype=np.int64)
    
    neg_features = []
    neg_count = int(len(pos_features) * negative_ratio)
    for seg in pos_segments:
        start_neg = seg[-1] + 1
        end_neg = seg[-1] + adjacent_steps
        if end_neg < X.shape[0] and np.all(y[start_neg:end_neg+1] == 0):
            neg_features.append(X[start_neg:end_neg+1].mean(axis=0))
        if len(neg_features) >= neg_count:
            break

    if len(neg_features) < neg_count:
        neg_idx = np.where(y == 0)[0]
        neg_segments = []
        if len(neg_idx) > 0:
            start = neg_idx[0]
            for i in range(1, len(neg_idx)):
                if neg_idx[i] != neg_idx[i-1] + 1:
                    neg_segments.append(np.arange(start, neg_idx[i-1]+1))
                    start = neg_idx[i]
            neg_segments.append(np.arange(start, neg_idx[-1]+1))
            for seg in neg_segments:
                if len(seg) >= adjacent_steps:
                    neg_features.append(X[seg[:adjacent_steps]].mean(axis=0))
                if len(neg_features) >= neg_count:
                    break
    neg_features = np.array(neg_features[:neg_count])
    neg_labels = np.zeros(len(neg_features), dtype=np.int64)
    features = np.concatenate([pos_features, neg_features], axis=0)
    labels = np.concatenate([pos_labels, neg_labels], axis=0)
    return features, labels
