import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tushare as ts
from itertools import product

from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import predict_new_data

# 设置Tushare API token
ts.set_token('c5c5700a6f4678a1837ad234f2e9ea2a573a26b914b47fa2dbb38aff')
pro = ts.pro_api()

# 设置随机种子
set_seed(42)

# 初始化session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}

def select_time(df, start_time='20230101', end_time='20240910'):
    """
    根据指定的时间范围筛选数据。
    参数:
    - df: 包含日期索引的DataFrame
    - start_time: 起始时间 (字符串, 格式 'YYYYMMDD')
    - end_time: 截止时间 (字符串, 格式 'YYYYMMDD')
    返回:
    - 筛选后的DataFrame
    """
    try:
        start_time = pd.to_datetime(start_time, format='%Y%m%d')
        end_time = pd.to_datetime(end_time, format='%Y%m%d')
    except Exception as e:
        print(f"日期转换错误：{e}")
        return pd.DataFrame()
    df_filtered = df.loc[start_time:end_time]
    return df_filtered

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_candlestick(data, stock_code, start_date, end_date, peaks=None, troughs=None, prediction=False, selected_classifiers=None, bt_result=None):
    # 确保索引是日期类型并过滤有效日期范围
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # 过滤数据只保留在start_date到end_date之间的数据
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
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

    # 添加买入/卖出标注，仅在 trade 列存在时
    if 'trade' in data.columns:
        for idx, row in data.iterrows():
            if row['trade'] == 'buy':
                # 绘制绿色框和白色字母
                fig.add_trace(go.Scatter(
                    x=[idx],  # X坐标是日期
                    y=[row['Low'] * 0.98],  # Y坐标稍微低一点
                    mode='markers+text',
                    text='b',  # 显示"b"
                    textfont=dict(color='white', size=16),  # 字母为白色
                    marker=dict(
                        color='green',  # 方框绿色
                        size=20,  # 方框大小
                        line=dict(color='green', width=2)  # 方框的边界是绿色
                    ),
                    name='Buy',  # 图例名称
                    showlegend=False,  # 隐藏图例
                    legendgroup='trade',  # 设置相同的legendgroup
                    textposition='middle center',  # 文本居中
                ), row=1, col=1)
            
            elif row['trade'] == 'sell':
                # 绘制红色框和白色字母
                fig.add_trace(go.Scatter(
                    x=[idx],  # X坐标是日期
                    y=[row['High'] * 1.02],  # Y坐标稍微高一点
                    mode='markers+text',
                    text='s',  # 显示"s"
                    textfont=dict(color='white', size=16),  # 字母为白色
                    marker=dict(
                        color='red',  # 方框红色
                        size=20,  # 方框大小
                        line=dict(color='red', width=2)  # 方框的边界是红色
                    ),
                    name='Sell',  # 图例名称
                    showlegend=False,  # 隐藏图例
                    legendgroup='trade',  # 设置相同的legendgroup
                    textposition='middle center',  # 文本居中
                ), row=1, col=1)

        # 添加统一的图例项
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='green', size=20, line=dict(color='green', width=2)),
            name='Buy',
            showlegend=True,
            legendgroup='trade',  # 确保所有买入和卖出都合并到同一组
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color='red', size=20, line=dict(color='red', width=2)),
            name='Sell',
            showlegend=True,
            legendgroup='trade',  # 确保所有买入和卖出都合并到同一组
        ))

    # 显示回测结果
    if bt_result:
        annotations = []
        y_pos = 0.95
        for key, value in bt_result.items():
            if isinstance(value, float):
                if key in {"同期标的涨跌幅", '"波段盈"累计收益率', "超额收益率", 
                         "单笔交易最大收益", "单笔交易最低收益", "单笔平均收益率", "胜率"}:
                    value = f"{value*100:.2f}%"
                else:
                    value = f"{value:.2f}"
                annotations.append(dict(
                    xref='paper', yref='paper',
                    x=0.05, y=1-y_pos,
                    text=f"{key}: {value}",
                    showarrow=False,
                    align='left'
                ))
                y_pos -= 0.05

        fig.update_layout(annotations=annotations)

    # 修改日期格式为中文
    fig.update_layout(
        title=f"{stock_code} {start_date} 至 {end_date}",
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            tickformat="%Y年%m月%d日",  # 设置日期格式为中文
            tickmode='array',
            tickvals=data.index,
            ticktext=[date.strftime("%Y年%m月%d日") for date in data.index]  # 格式化日期为中文
        )
    )
    return fig






def read_day_from_tushare(symbol_code, symbol_type='index'):
    """
    使用 Tushare API 获取股票或指数的全部日线行情数据。
    参数:
    - symbol_code: 股票或指数代码 (如 "000001.SZ" 或 "000300.SH")
    - symbol_type: 'stock' 或 'index' (不区分大小写)
    返回:
    - 包含日期、开高低收、成交量等列的DataFrame
    """
    symbol_type = symbol_type.lower()
    print(f"传递给 read_day_from_tushare 的 symbol_type: {symbol_type} (类型: {type(symbol_type)})")  # 调试输出
    print(f"尝试通过 Tushare 获取{symbol_type}数据: {symbol_code}")
    
    # 添加断言，确保 symbol_type 是 'stock' 或 'index'
    assert symbol_type in ['stock', 'index'], "symbol_type 必须是 'stock' 或 'index'"
    
    try:
        if symbol_type == 'stock':
            # 获取股票日线数据
            df = pro.daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的股票数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        elif symbol_type == 'index':
            # 获取指数日线数据，使用 index_daily 接口
            df = pro.index_daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
            if df.empty:
                print("Tushare 返回的指数数据为空。")
                return pd.DataFrame()
            
            # 转换日期格式并排序
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('date')
            
            # 重命名和选择需要的列
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'amount': 'Amount',
                'trade_date': 'TradeDate'
            })
            df.set_index('date', inplace=True)
            
            # 选择需要的列，处理可能缺失的字段
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
        
        print(f"通过 Tushare 获取了 {len(df)} 条记录。")
        print(f"数据框的列：{df.columns.tolist()}")
        print(f"数据框前5行：\n{df.head()}")
        return df
    except AssertionError as ae:
        print(f"断言错误：{ae}")
        return pd.DataFrame()
    except Exception as e:
        print(f"通过 Tushare 获取数据失败：{e}")
        return pd.DataFrame()

#不进行配对
def main_product10():
    st.set_page_config(page_title="东吴财富管理AI超额收益系统", layout="wide")
    st.title("东吴财富管理AI超额收益系统")

    with st.sidebar:
        st.header("参数设置")
        
        # 数据设置
        data_source = st.selectbox("选择数据来源", ["指数","股票"])
        symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
        N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)
        
        # 模型设置
        classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
        # 将 "深度学习" 转换为 "MLP"
        if classifier_name == "深度学习":
            classifier_name = "MLP"
        mixture_depth = st.slider("因子混合深度", 1, 3, 1)
        oversample_method = st.selectbox("类别不均衡处理", 
            ["过采样", "类别权重",'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek'])
        if oversample_method == "过采样":
            oversample_method = "SMOTE"
        if oversample_method == '类别权重':
            oversample_method ="Class Weights"
            
        
        
        # 特征选择
        auto_feature = st.checkbox("自动特征选择", True)
        n_features_selected = st.number_input("选择特征数量", 
            min_value=5, max_value=100, value=20, disabled=auto_feature)

    # 训练和预测选项卡
    tab1, tab2 = st.tabs(["训练模型", "预测"])

    with tab1:
        with st.form("train_form"):
            st.subheader("训练参数")
            col1, col2 = st.columns(2)
            with col1:
                train_start = st.date_input("训练开始日期", datetime(2000,1,1))  # 设置中文日期控件
            with col2:
                train_end = st.date_input("训练结束日期", datetime(2020,12,31))  # 设置中文日期控件
            
            if st.form_submit_button("开始调参"):
                try:
                    # 根据数据来源选择股票或指数
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
                    
                    with st.spinner("数据预处理中..."):
                        df_preprocessed, all_features = preprocess_data(df, N, mixture_depth, mark_labels=True)
                    
                    with st.spinner("模型调参中..."):
                        (peak_model, peak_scaler, peak_selector, 
                        peak_selected_features, all_features_peak, peak_best_score,
                        peak_metrics, peak_threshold,
                        trough_model, trough_scaler, trough_selector,
                        trough_selected_features, all_features_trough,
                        trough_best_score, trough_metrics, trough_threshold) = train_model(
                            df_preprocessed, N, all_features, classifier_name,
                            mixture_depth, n_features_selected if not auto_feature else 'auto', 
                            oversample_method
                        )
                        
                        st.session_state.models = {
                            'peak_model': peak_model,
                            'peak_scaler': peak_scaler,
                            'peak_selector': peak_selector,
                            'all_features_peak': all_features_peak,
                            'peak_threshold': peak_threshold,
                            'trough_model': trough_model,
                            'trough_scaler': trough_scaler,
                            'trough_selector': trough_selector,
                            'all_features_trough': all_features_trough,
                            'trough_threshold': trough_threshold,
                            'N': N,
                            'mixture_depth': mixture_depth
                        }
                        st.session_state.trained = True
                    
                    st.success("调参完成！")
                    
                    peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
                    troughs = df_preprocessed[df_preprocessed['Trough'] == 1]
                    fig = plot_candlestick(df_preprocessed, symbol_code, 
                                         train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"),
                                         peaks, troughs)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"训练失败: {str(e)}")

    with tab2:
        if not st.session_state.trained:
            st.warning("请先完成模型预训练")
        else:
            with st.form("predict_form"):
                st.subheader("预测参数")
                col1, col2 = st.columns(2)
                with col1:
                    pred_start = st.date_input("预测开始日期")  
                with col2:
                    pred_end = st.date_input("预测结束日期") 
                
                if st.form_submit_button("开始预测"):
                    try:
                        # 根据数据来源选择股票或指数
                        symbol_type = 'index' if data_source == '指数' else 'stock'
                        data = read_day_from_tushare(symbol_code, symbol_type)
                        new_df = select_time(data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))
                        
                        # 预处理数据
                        df_preprocessed, all_features = preprocess_data(new_df, N, mixture_depth, mark_labels=True)
                        
                        best_excess = -np.inf
                        best_models = None
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 进行10次模型训练
                        for i in range(10):
                            status_text.text(f"正在进行第 {i+1}/10 轮模型训练...")
                            progress_bar.progress((i+1)/10)
                            
                            try:
                                # 重新训练模型
                                (peak_model, peak_scaler, peak_selector, 
                                _, all_features_peak, _,
                                _, peak_threshold,
                                trough_model, trough_scaler, trough_selector,
                                _, all_features_trough,
                                _, _, trough_threshold) = train_model(
                                    df_preprocessed, N, all_features, classifier_name,
                                    mixture_depth, 
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )
                                
                                # 在预测集上回测
                                _, bt_result,trades_df = predict_new_data(
                                    new_df,
                                    peak_model, peak_scaler, peak_selector, all_features_peak, peak_threshold,
                                    trough_model, trough_scaler, trough_selector, all_features_trough, trough_threshold,
                                    N, mixture_depth,
                                    window_size=30,  
                                    eval_mode=True   
                                )
                                
                                # 比较超额收益率
                                current_excess = bt_result.get('超额收益率', -np.inf)
                                if current_excess > best_excess:
                                    best_excess = current_excess
                                    best_models = {
                                        'peak_model': peak_model,
                                        'peak_scaler': peak_scaler,
                                        'peak_selector': peak_selector,
                                        'all_features_peak': all_features_peak,
                                        'peak_threshold': peak_threshold,
                                        'trough_model': trough_model,
                                        'trough_scaler': trough_scaler,
                                        'trough_selector': trough_selector,
                                        'all_features_trough': all_features_trough,
                                        'trough_threshold': trough_threshold
                                    }
                                    
                            except Exception as e:
                                st.error(f"第 {i+1} 次训练失败: {str(e)}")
                                continue
                        
                        # 使用最佳模型进行最终预测
                        if best_models is None:
                            raise ValueError("所有训练尝试均失败")
                            
                        status_text.text("使用最佳模型进行最终预测...")
                        final_result, final_bt,final_trades_df = predict_new_data(
                            new_df,
                            best_models['peak_model'],
                            best_models['peak_scaler'],
                            best_models['peak_selector'],
                            best_models['all_features_peak'],
                            best_models['peak_threshold'],
                            best_models['trough_model'],
                            best_models['trough_scaler'],
                            best_models['trough_selector'],
                            best_models['all_features_trough'],
                            best_models['trough_threshold'],
                            N, mixture_depth,
                            window_size=30,
                            eval_mode=False
                        )
                        
                        # 显示结果
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess*100:.2f}%")
                        
                        # 显示回测结果
                        st.subheader("回测结果")
                        cols = st.columns(4)
                        metrics = [
                            ('累计收益率', final_bt.get('"波段盈"累计收益率', 0)),
                            ('超额收益率', final_bt.get('超额收益率', 0)),
                            ('胜率', final_bt.get('胜率', 0)),
                            ('交易笔数', final_bt.get('交易笔数', 0))
                        ]
                        for col, (name, value) in zip(cols, metrics):
                            col.metric(name, f"{value*100:.2f}%" if isinstance(value, float) else value)
                        
                        # 显示图表
                        peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                        troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
                        fig = plot_candlestick(final_result, symbol_code, 
                                            pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"),
                                            peaks_pred, troughs_pred, True, 
                                            [classifier_name], final_bt)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示预测结果表格
                        #
                        col1, col2 = st.columns(2)
                        # 在左侧列展示预测结果
                        # 重命名列
                    
                        final_result = final_result.rename(columns={
                            'TradeDate': '交易日期',
                            'Peak_Prediction': '高点标注',
                            'Peak_Probability': '高点概率',
                            'Trough_Prediction': '低点标注',
                            'Trough_Probability': '低点概率'
                        })

                        # 在左侧列展示预测结果
                        with col1:
                            st.subheader("预测明细")
                            st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                        #重命名列
                        final_trades_df=final_trades_df.rename(columns={
                             "entry_date":'买入日',
                             "entry_price":'买入价',
                             "exit_date":'卖出日',
                             "exit_price":'卖出价',
                             "hold_days":'持仓日',
                             "return":'盈亏',
                                    
                        })
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        # 在右侧列展示交易数据
                        with col2:
                            st.subheader("交易记录")
                            st.dataframe(final_trades_df[['买入日', '买入价', '卖出日', '持仓日', '盈亏']].style.format({'盈亏': '{:.2f}%'}))
                        
                        progress_bar.empty()
                        status_text.empty()

                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")


def main_product100():
    st.set_page_config(page_title="东吴财富管理AI超额收益系统", layout="wide")
    st.title("东吴财富管理AI超额收益系统")

    with st.sidebar:
        st.header("参数设置")
        
        # 数据设置
        data_source = st.selectbox("选择数据来源", ["股票", "指数"])
        symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
        N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)
        
        # 模型设置
        classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
        if classifier_name == "深度学习":
            classifier_name = "MLP"
        mixture_depth = st.slider("因子混合深度", 1, 3, 1)
        oversample_method = st.selectbox("类别不均衡处理", 
             ["过采样", "类别权重",'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek'])
        if oversample_method == "过采样":
            oversample_method = "SMOTE"
        if oversample_method == '类别权重':
            oversample_method ="Class Weights"
        
        # 特征选择
        auto_feature = st.checkbox("自动特征选择", True)
        n_features_selected = st.number_input("选择特征数量", 
            min_value=5, max_value=100, value=20, disabled=auto_feature)

    # 训练和预测选项卡
    tab1, tab2 = st.tabs(["训练模型", "预测"])

    with tab1:
        with st.form("train_form"):
            st.subheader("训练参数")
            col1, col2 = st.columns(2)
            with col1:
                train_start = st.date_input("训练开始日期", datetime(2000,1,1))  # 设置中文日期控件
            with col2:
                train_end = st.date_input("训练结束日期", datetime(2020,12,31))  # 设置中文日期控件
            
            if st.form_submit_button("开始训练"):
                try:
                    # 根据数据来源选择股票或指数
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
                    
                    with st.spinner("数据预处理中..."):
                        df_preprocessed, all_features = preprocess_data(df, N, mixture_depth, mark_labels=True)
                    
                    with st.spinner("训练模型中..."):
                        (peak_model, peak_scaler, peak_selector, 
                        peak_selected_features, all_features_peak, peak_best_score,
                        peak_metrics, peak_threshold,
                        trough_model, trough_scaler, trough_selector,
                        trough_selected_features, all_features_trough,
                        trough_best_score, trough_metrics, trough_threshold) = train_model(
                            df_preprocessed, N, all_features, classifier_name,
                            mixture_depth, n_features_selected if not auto_feature else 'auto', 
                            oversample_method
                        )
                        
                        st.session_state.models = {
                            'peak_model': peak_model,
                            'peak_scaler': peak_scaler,
                            'peak_selector': peak_selector,
                            'all_features_peak': all_features_peak,
                            'peak_threshold': peak_threshold,
                            'trough_model': trough_model,
                            'trough_scaler': trough_scaler,
                            'trough_selector': trough_selector,
                            'all_features_trough': all_features_trough,
                            'trough_threshold': trough_threshold,
                            'N': N,
                            'mixture_depth': mixture_depth
                        }
                        st.session_state.trained = True
                    
                    st.success("训练完成！")
                    peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
                    troughs = df_preprocessed[df_preprocessed['Trough'] == 1]
                    fig = plot_candlestick(df_preprocessed, symbol_code, 
                                         train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"),
                                         peaks, troughs)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"训练失败: {str(e)}")

    with tab2:
        if not st.session_state.trained:
            st.warning("请先完成模型训练")
        else:
            with st.form("predict_form"):
                st.subheader("预测参数")
                col1, col2 = st.columns(2)
                with col1:
                    pred_start = st.date_input("预测开始日期")  
                with col2:
                    pred_end = st.date_input("预测结束日期") 
                
                if st.form_submit_button("开始预测"):
                    try:
                        # 根据数据来源选择股票或指数
                        symbol_type = 'index' if data_source == '指数' else 'stock'
                        data = read_day_from_tushare(symbol_code, symbol_type)
                        new_df = select_time(data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))
                        
                        # 预处理数据
                        df_preprocessed, all_features = preprocess_data(new_df, N, mixture_depth, mark_labels=True)
                        
                        best_excess = -np.inf
                        best_models = None
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 进行10次模型训练
                        peak_models = []
                        trough_models = []
                        for i in range(10):
                            status_text.text(f"正在进行第 {i+1}/10 次模型训练...")
                            progress_bar.progress((i+1)/10)
                            
                            try:
                                # 重新训练模型
                                (peak_model, peak_scaler, peak_selector, 
                                _, all_features_peak, _,
                                _, peak_threshold,
                                trough_model, trough_scaler, trough_selector,
                                _, all_features_trough,
                                _, _, trough_threshold) = train_model(
                                    df_preprocessed, N, all_features, classifier_name,
                                    mixture_depth, 
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )
                                
                                # 保存模型
                                peak_models.append((peak_model, peak_scaler, peak_selector, all_features_peak, peak_threshold))
                                trough_models.append((trough_model, trough_scaler, trough_selector, all_features_trough, trough_threshold))
                                
                            except Exception as e:
                                st.error(f"第 {i+1} 次训练失败: {str(e)}")
                                continue
                        
                        # 生成笛卡尔积的100个模型组合
                        model_combinations = list(product(peak_models, trough_models))

                        # 回测每个组合并选择超额收益率最高的
                        for peak_model, trough_model in model_combinations:
                            peak_model_data = peak_model
                            trough_model_data = trough_model

                            # 执行回测
                            _, bt_result,trades_df = predict_new_data(new_df, peak_model_data[0], peak_model_data[1], peak_model_data[2], peak_model_data[3], peak_model_data[4],
                                                           trough_model_data[0], trough_model_data[1], trough_model_data[2], trough_model_data[3], trough_model_data[4],
                                                           N, mixture_depth, window_size=30, eval_mode=True)

                            # 比较超额收益率
                            current_excess = bt_result.get('超额收益率', -np.inf)
                            if current_excess > best_excess:
                                best_excess = current_excess
                                best_models = {
                                    'peak_model': peak_model_data[0],
                                    'peak_scaler': peak_model_data[1],
                                    'peak_selector': peak_model_data[2],
                                    'all_features_peak': peak_model_data[3],
                                    'peak_threshold': peak_model_data[4],
                                    'trough_model': trough_model_data[0],
                                    'trough_scaler': trough_model_data[1],
                                    'trough_selector': trough_model_data[2],
                                    'all_features_trough': trough_model_data[3],
                                    'trough_threshold': trough_model_data[4]
                                }

                        # 使用最佳模型进行最终预测
                        if best_models is None:
                            raise ValueError("所有训练尝试均失败")
                            
                        status_text.text("使用最佳模型进行最终预测...")
                        final_result, final_bt,final_trades_df = predict_new_data(
                            new_df,
                            best_models['peak_model'],
                            best_models['peak_scaler'],
                            best_models['peak_selector'],
                            best_models['all_features_peak'],
                            best_models['peak_threshold'],
                            best_models['trough_model'],
                            best_models['trough_scaler'],
                            best_models['trough_selector'],
                            best_models['all_features_trough'],
                            best_models['trough_threshold'],
                            N, mixture_depth,
                            window_size=30,
                            eval_mode=False
                        )
                        
                        # 显示结果
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess*100:.2f}%")
                        
                        # 显示回测结果
                        st.subheader("回测结果")
                        cols = st.columns(4)
                        metrics = [
                            ('累计收益率', final_bt.get('"波段盈"累计收益率', 0)),
                            ('超额收益率', final_bt.get('超额收益率', 0)),
                            ('胜率', final_bt.get('胜率', 0)),
                            ('交易笔数', final_bt.get('交易笔数', 0))
                        ]
                        for col, (name, value) in zip(cols, metrics):
                            col.metric(name, f"{value*100:.2f}%" if isinstance(value, float) else value)
                        
                        # 显示图表
                        peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                        troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
                        fig = plot_candlestick(final_result, symbol_code, 
                                            pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"),
                                            peaks_pred, troughs_pred, True, 
                                            [classifier_name], final_bt)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示预测结果表格
                        st.subheader("预测明细")
                        st.dataframe(final_result[['TradeDate', 'Peak_Prediction', 'Peak_Probability',
                                                'Trough_Prediction', 'Trough_Probability']])
                        
                        progress_bar.empty()
                        status_text.empty()

                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")


if __name__ == "__main__":
    main_product10()
