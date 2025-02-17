# predict.py
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from skorch import NeuralNetClassifier
from backtest import build_daily_equity_curve,build_trades_from_signals,backtest_results
from models import  TransformerClassifier
from plotly.subplots import make_subplots
import plotly.graph_objs as go

#绘图函数
def plot_candlestick_plotly(self, data, stock_code, start_date, end_date, peaks=None, troughs=None, prediction=False):
        if prediction and self.selected_classifiers:
            classifiers_str = ", ".join(self.selected_classifiers)
            title = f"{stock_code} {start_date} 至 {end_date} 基础模型: {classifiers_str}"
        else:
            title = f"{stock_code} {start_date} 至 {end_date}"

        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"data.index 无法转换为日期格式: {e}")
        data.index = data.index.strftime('%Y-%m-%d')

        if peaks is not None and not peaks.empty:
            if not isinstance(peaks.index, pd.DatetimeIndex):
                try:
                    peaks.index = pd.to_datetime(peaks.index)
                except Exception as e:
                    raise ValueError(f"peaks.index 无法转换为日期格式: {e}")
            peaks.index = peaks.index.strftime('%Y-%m-%d')

        if troughs is not None and not troughs.empty:
            if not isinstance(troughs.index, pd.DatetimeIndex):
                try:
                    troughs.index = pd.to_datetime(troughs.index)
                except Exception as e:
                    raise ValueError(f"troughs.index 无法转换为日期格式: {e}")
            troughs.index = troughs.index.strftime('%Y-%m-%d')

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "candlestick"}],[{"type": "bar"}]]
        )

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=stock_code,
            increasing=dict(line=dict(color='red')),
            decreasing=dict(line=dict(color='green')),
            hoverinfo='x+y+text',
        ), row=1, col=1)

        if 'Volume' in data.columns:
            volume_colors = ['red' if row['Close'] > row['Open'] else 'green' for _, row in data.iterrows()]
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=volume_colors,
                name='成交量',
                hoverinfo='x+y'
            ), row=2, col=1)

        if peaks is not None and not peaks.empty:
            marker_y_peaks = peaks['High'] * 1.02
            marker_x_peaks = peaks.index
            color_peak = 'green'
            label_peak = '局部高点' if not prediction else '预测高点'
            fig.add_trace(go.Scatter(
                x=marker_x_peaks,
                y=marker_y_peaks,
                mode='text',
                text='W',
                textfont=dict(color=color_peak, size=20),
                name=label_peak
            ), row=1, col=1)

        if troughs is not None and not troughs.empty:
            marker_y_troughs = troughs['Low'] * 0.98
            marker_x_troughs = troughs.index
            color_trough = 'red'
            label_trough = '局部低点' if not prediction else '预测低点'
            fig.add_trace(go.Scatter(
                x=marker_x_troughs,
                y=marker_y_troughs,
                mode='text',
                text='D',
                textfont=dict(color=color_trough, size=20),
                name=label_trough
            ), row=1, col=1)

        # 显示每个回测结果指标在图表右侧
        if self.bt_result is not None:
            y_position = 1  # 起始Y位置
            for key, value in self.bt_result.items():
                if isinstance(value, float):  # 确保只显示数字类型
                    if key in {"同期标的涨跌幅", '"波段盈"累计收益率', "超额收益率", 
                               "单笔交易最大收益", "单笔交易最低收益", "单笔平均收益率", "胜率"}:
                        value_display = f"{value*100:.2f}%"  # 百分比显示
                    else:
                        value_display = f"{value:.2f}"

                    # 添加每个回测结果到图表的右侧
                    fig.add_annotation(
                        text=f"{key}: {value_display}",
                        xref="paper", yref="paper",
                        x=1.12,  # 让文本左对齐，适当调整
                        y=0.8 - y_position * 0.06,  # 控制Y位置，使其分段排列
                        showarrow=False,
                        align="left",  # 设为左对齐
                        bordercolor="black",
                        borderwidth=1,
                        bgcolor="white",
                        opacity=0.8
                    )
                    y_position += 1  # 向下偏移下一行

        fig.update_layout(
            title=title,
            xaxis=dict(
                title="日期",
                type="category",
                tickangle=45,
                tickmode="auto",
                nticks=10
            ),
            xaxis2=dict(
                title="日期",
                type="category",
                tickangle=45,
                tickmode="auto",
                nticks=10
            ),
            yaxis_title="价格",
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            height=800,
            font=dict(
                family="Microsoft YaHei, SimHei",
                size=14,
                color="black"
            )
        )

        html = fig.to_html(include_plotlyjs='cdn')
        html = html.replace('</head>', '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>')
        html = html.replace(
            '<body>',
            '<body><script src="https://cdn.plot.ly/locale/zh-cn.js"></script><script>Plotly.setPlotConfig({locale: "zh-CN"});</script>'
        )
        return html
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

    # 回测：生成交易信号并计算回测结果
    signal_df = get_trade_signal(data_preprocessed)
    bt_result = backtest_results(data_preprocessed, signal_df, initial_capital=1_000_000)
    print("回测结果：", bt_result)
    return data_preprocessed, bt_result

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


    

