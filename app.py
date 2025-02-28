import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
from itertools import product
import pickle  # 用于模型的序列化/反序列化

from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick

# 设置随机种子
set_seed(42)

# 初始化 session_state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = None  # 存储"最佳模型"的组合

def load_custom_css():
    """
    一段自定义 CSS，主要用于微调列间距和标签对齐方式。
    """
    custom_css = """
    <style>
    /* 调整列间距 */
    .strategy-row {
        margin-bottom: 8px;
    }
    /* 强制统一对齐 */
    .strategy-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def main_product():
    """
    1) Tab1: 训练模型（可视化训练集，点击“开始训练”才进行模型训练）。
    2) Tab2: 直接在此重新训练或搜索最佳模型并预测，可下载保存最佳模型。
    3) Tab3: 上传已有模型文件（.pkl）进行预测，参数设置布局与Tab2对齐。
    """
    st.set_page_config(page_title="东吴秀享AI超额收益系统", layout="wide")
    st.title("东吴秀享AI超额收益系统")

    # ========== 侧边栏参数设置 ========== #
    with st.sidebar:
        st.header("参数设置")

        # 数据设置分组
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)

        # 模型设置分组
        with st.expander("模型设置", expanded=True):
            classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            if classifier_name == "深度学习":
                classifier_name = "MLP"  # 原逻辑：深度学习 -> MLP
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_method = st.selectbox(
                "类别不均衡处理", 
                ["过采样", "类别权重", 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek']
            )
            # 将“过采样”映射为“SMOTE”、将“类别权重”映射为“Class Weights”
            if oversample_method == "过采样":
                oversample_method = "SMOTE"
            if oversample_method == '类别权重':
                oversample_method = "Class Weights"

            # 是否使用“10×10 笛卡尔积”组合最佳模型 的开关
            use_best_combo = st.checkbox(
                "精细化筛选模型", 
                value=False, 
                help="若开启，则将训练10个峰模型与10个谷模型的所有组合(100)进行回测，选出收益最高的组合。"
            )

        # 特征设置分组
        with st.expander("特征设置", expanded=True):
            auto_feature = st.checkbox("自动特征选择", True)
            n_features_selected = st.number_input(
                "选择特征数量", 
                min_value=5, max_value=100, value=20, 
                disabled=auto_feature
            )

        
    # ========== 三个选项卡 ========== #
    tab1, tab2, tab3 = st.tabs(["训练模型", "预测", "上传模型预测"])

    # =======================================
    #         Tab1: 训练模型
    # =======================================
    with tab1:
        st.subheader("训练参数")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("训练开始日期", datetime(2000, 1, 1))
        with col2:
            train_end = st.date_input("训练结束日期", datetime(2020, 12, 31))

        # --- 点击“开始训练”才执行 train_model() ---
        if st.button("开始训练"):
            try:
                with st.spinner("数据预处理中..."):
                    # 读取相同区间的数据
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))

                    # 做标注
                    df_preprocessed, all_features = preprocess_data(
                        df, 
                        N, 
                        mixture_depth, 
                        mark_labels=True
                    )

                with st.spinner("模型训练中..."):
                    # 正式调用 train_model()
                    (peak_model, peak_scaler, peak_selector,
                     peak_selected_features, all_features_peak, peak_best_score,
                     peak_metrics, peak_threshold,
                     trough_model, trough_scaler, trough_selector,
                     trough_selected_features, all_features_trough,
                     trough_best_score, trough_metrics, trough_threshold) = train_model(
                        df_preprocessed,
                        N,
                        all_features,
                        classifier_name,
                        mixture_depth,
                        n_features_selected if not auto_feature else 'auto',
                        oversample_method
                    )

                    # 保存模型到 session_state
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

                # 训练完成后，再画一次波峰波谷图
                peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
                troughs = df_preprocessed[df_preprocessed['Trough'] == 1]
                fig = plot_candlestick(
                    df_preprocessed,
                    symbol_code,
                    train_start.strftime("%Y%m%d"),
                    train_end.strftime("%Y%m%d"),
                    peaks=peaks,
                    troughs=troughs
                )
                st.plotly_chart(fig, use_container_width=True, key="chart1")

            except Exception as e:
                st.error(f"训练失败: {str(e)}")

        # ---------- (A) 可视化训练集：仅做数据预处理(标记峰/谷) ----------
        try:
            st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
            symbol_type = 'index' if data_source == '指数' else 'stock'
            data = read_day_from_tushare(symbol_code, symbol_type)
            df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))

            df_preprocessed, all_features = preprocess_data(
                df, 
                N, 
                mixture_depth,
                mark_labels=True
            )

            peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
            troughs = df_preprocessed[df_preprocessed['Trough'] == 1]

            fig = plot_candlestick(
                df_preprocessed,
                symbol_code,
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                peaks=peaks,
                troughs=troughs
            )
            st.plotly_chart(fig, use_container_width=True, key="chart2")

        except Exception as e:
            st.warning(f"可视化失败: {e}")

    # =======================================
    #    Tab2: 自动训练 + 预测 + 保存模型
    # =======================================
    with tab2:
        if not st.session_state.get('trained', False):
            st.warning("请先完成模型训练")
        else:
            st.subheader("预测参数")

            # 1) 预测起止日期
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2:
                pred_end = st.date_input("预测结束日期", datetime.now())

            # 2) 策略选择
            with st.expander("策略选择", expanded=False):
                load_custom_css()

                # 示例：追涨策略
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab2"
                    )

                # 示例：止损策略
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab2")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab2"
                    )

                # 示例：高点需创X日新高
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab2")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab2"
                    )

            if st.button("开始预测"):
                try:
                    # 读取 & 预处理预测数据
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    new_df = select_time(data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))

                    df_preprocessed, all_features = preprocess_data(
                        new_df,
                        st.session_state.models.get('N', 30),  # 从已训练模型中取 N
                        st.session_state.models.get('mixture_depth', 1),
                        mark_labels=True
                    )

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # (单峰+单谷) 或 (10×10 笛卡尔积)
                    best_excess = -np.inf
                    best_models = None
                    final_result = None
                    final_bt = {}
                    final_trades_df = pd.DataFrame()

                    enable_chase_val = enable_chase
                    enable_stop_loss_val = enable_stop_loss
                    enable_change_signal_val = enable_change_signal

                    # 为了避免冲突，重新获取当前策略参数
                    n_buy_val = st.session_state.get("n_buy_tab2", 10)
                    n_sell_val = st.session_state.get("n_sell_tab2", 10)
                    n_newhigh_val = st.session_state.get("n_newhigh_tab2", 60)

                    # ------------------ 不使用笛卡尔积 ------------------
                    if not use_best_combo:
                        # (单峰+单谷) 重复训练10次，选最佳
                        for i in range(10):
                            status_text.text(f"正在进行第 {i+1}/10 次模型训练...")
                            progress_bar.progress((i+1) / 10)

                            try:
                                (peak_model, peak_scaler, peak_selector,
                                 _, all_features_peak, _,
                                 _, peak_threshold,
                                 trough_model, trough_scaler, trough_selector,
                                 _, all_features_trough,
                                 _, _, trough_threshold) = train_model(
                                    df_preprocessed,
                                    st.session_state.models['N'],
                                    all_features,
                                    classifier_name,
                                    st.session_state.models['mixture_depth'],
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )

                                # 回测
                                _, bt_result, _ = predict_new_data(
                                    new_df,
                                    peak_model, peak_scaler, peak_selector, all_features_peak, peak_threshold,
                                    trough_model, trough_scaler, trough_selector, all_features_trough, trough_threshold,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=30,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
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

                        if best_models is None:
                            raise ValueError("所有训练尝试均失败，无法找到可用模型。")

                        # 用最佳模型做最终预测
                        status_text.text("使用最佳模型进行最终预测...")
                        final_result, final_bt, final_trades_df = predict_new_data(
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

                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=30,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # ------------------ 使用笛卡尔积(10×10) ------------------
                    else:
                        peak_models = []
                        trough_models = []

                        # 先各训练10次峰模型和谷模型
                        for i in range(10):
                            status_text.text(f"正在训练第 {i+1}/10 个峰+谷模型...")
                            progress_bar.progress((i+1) / 10)

                            try:
                                (peak_model, peak_scaler, peak_selector,
                                 _, all_features_peak, _,
                                 _, peak_threshold,
                                 trough_model, trough_scaler, trough_selector,
                                 _, all_features_trough,
                                 _, _, trough_threshold) = train_model(
                                    df_preprocessed,
                                    st.session_state.models['N'],
                                    all_features,
                                    classifier_name,
                                    st.session_state.models['mixture_depth'],
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )
                                peak_models.append((
                                    peak_model, peak_scaler, peak_selector, 
                                    all_features_peak, peak_threshold
                                ))
                                trough_models.append((
                                    trough_model, trough_scaler, trough_selector, 
                                    all_features_trough, trough_threshold
                                ))
                            except Exception as e:
                                st.error(f"第 {i+1} 次训练失败: {str(e)}")
                                continue

                        # 笛卡尔积组合
                        model_combinations = list(product(peak_models, trough_models))
                        total_combos = len(model_combinations)  # 预期 10×10=100
                        for idx, (peak_model_data, trough_model_data) in enumerate(model_combinations):
                            combo_progress = (idx + 1) / total_combos
                            status_text.text(f"正在回测第 {idx+1}/{total_combos} 个组合...")
                            progress_bar.progress(combo_progress)

                            pm, ps, psel, paf, pth = peak_model_data
                            tm, ts, tsel, taf, tth = trough_model_data

                            try:
                                _, bt_result, _ = predict_new_data(
                                    new_df,
                                    pm, ps, psel, paf, pth,
                                    tm, ts, tsel, taf, tth,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=30,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
                                current_excess = bt_result.get('超额收益率', -np.inf)
                                if current_excess > best_excess:
                                    best_excess = current_excess
                                    best_models = {
                                        'peak_model': pm,
                                        'peak_scaler': ps,
                                        'peak_selector': psel,
                                        'all_features_peak': paf,
                                        'peak_threshold': pth,

                                        'trough_model': tm,
                                        'trough_scaler': ts,
                                        'trough_selector': tsel,
                                        'all_features_trough': taf,
                                        'trough_threshold': tth
                                    }
                            except:
                                # 忽略某些异常
                                continue

                        if best_models is None:
                            raise ValueError("所有组合均测试失败，无法完成预测。")

                        # 用最佳组合做最终预测
                        status_text.text("使用最佳组合进行最终预测...")
                        final_result, final_bt, final_trades_df = predict_new_data(
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

                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=30,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # --- 将“最佳模型”写入 session_state，可后续上传下载 ---
                    if best_models is not None:
                        st.session_state.best_models = best_models
                        # 去掉原先的 expander，直接给出一个“保存最佳模型”按钮
                        st.download_button(
                            label="保存最佳模型",
                            data=pickle.dumps(best_models),
                            file_name="best_models.pkl",
                            mime="application/octet-stream"
                        )

                    # ========== 回测结果展示 ========== #
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('"波段盈"累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',     final_bt.get('年化夏普比率', 0)),
                    ]

                    # 前三列
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # 后三列
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # K线图：标注预测的峰/谷
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]

                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True,
                        selected_classifiers=[classifier_name],
                        bt_result=final_bt
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart3")

                    # 表格展示
                    col_left, col_right = st.columns(2)

                    # 预测明细
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    # 交易记录
                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[[
                                    '买入日', '买入原因', '买入价', 
                                    '卖出日', '卖出原因', '卖出价', 
                                    '持仓日', '盈亏'
                                ]].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")

                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

    # =======================================
    #    Tab3: 上传模型文件，独立进行预测
    # =======================================
    with tab3:
        st.subheader("上传模型文件（.pkl）并预测")
        st.markdown("在此页面可以上传之前已保存的最佳模型文件，直接进行预测。")

        # 1) 上传模型
        uploaded_file = st.file_uploader("选择本地模型文件进行预测：", type=["pkl"])
        if uploaded_file is not None:
            with st.spinner("正在加载模型..."):
                best_models_loaded = pickle.load(uploaded_file)
                st.session_state.best_models = best_models_loaded
                # 一旦加载了模型，就视为已“trained”，可以进行预测
                st.session_state.trained = True
            st.success("已成功加载本地模型，可进行预测！")

        if not st.session_state.trained or (st.session_state.best_models is None):
            st.warning("请先上传模型文件，或前往其他页面训练并保存模型。")
        else:
            st.markdown("### 预测参数")
            # 与 Tab2 布局一致
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start_up = st.date_input("预测开始日期(上传模型Tab)", datetime(2021, 1, 1))
            with col_date2:
                pred_end_up = st.date_input("预测结束日期(上传模型Tab)", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()

                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab3")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab3"
                    )

                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab3")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab3"
                    )

                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab3")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab3"
                    )

            # 2) 点击预测
            if st.button("开始预测(上传模型Tab)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    new_df = select_time(data, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

                    # 这里的 N、mixture_depth 可从 st.session_state.models 中获取，
                    # 或直接从 best_models 里看看是否存了 N, mixture_depth。如果没有就设置默认值
                    N_val = st.session_state.models.get('N', 30)
                    mixture_val = st.session_state.models.get('mixture_depth', 1)

                    df_preprocessed, all_features = preprocess_data(
                        new_df,
                        N_val,
                        mixture_val,
                        mark_labels=True
                    )

                    # 执行预测
                    final_result, final_bt, final_trades_df = predict_new_data(
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

                        N_val,
                        mixture_val,
                        window_size=30,
                        eval_mode=False,
                        N_buy=n_buy_up,
                        N_sell=n_sell_up,
                        N_newhigh=n_newhigh_up,
                        enable_chase=enable_chase_up,
                        enable_stop_loss=enable_stop_loss_up,
                        enable_change_signal=enable_change_signal_up,
                    )

                    st.success("预测完成！（使用已上传模型）")

                    # 回测结果展示
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('"波段盈"累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',     final_bt.get('年化夏普比率', 0)),
                    ]

                    # 前三列
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # 后三列
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # K线图：标注预测的峰/谷
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]

                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start_up.strftime("%Y%m%d"),
                        pred_end_up.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart_upload_tab")

                    # 显示预测明细 & 交易记录
                    col_left, col_right = st.columns(2)
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[[
                                    '买入日', '买入原因', '买入价', 
                                    '卖出日', '卖出原因', '卖出价', 
                                    '持仓日', '盈亏'
                                ]].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

if __name__ == "__main__":
    main_product()
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
from itertools import product
import pickle  # 用于模型的序列化/反序列化

from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick

# 设置随机种子
set_seed(42)

# 初始化 session_state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = None  # 存储"最佳模型"的组合

def load_custom_css():
    """
    一段自定义 CSS，主要用于微调列间距和标签对齐方式。
    """
    custom_css = """
    <style>
    /* 调整列间距 */
    .strategy-row {
        margin-bottom: 8px;
    }
    /* 强制统一对齐 */
    .strategy-label {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def main_product():
    """
    1) Tab1: 训练模型（可视化训练集，点击“开始训练”才进行模型训练）。
    2) Tab2: 直接在此重新训练或搜索最佳模型并预测，可下载保存最佳模型。
    3) Tab3: 上传已有模型文件（.pkl）进行预测，参数设置布局与Tab2对齐。
    """
    st.set_page_config(page_title="东吴秀享AI超额收益系统", layout="wide")
    st.title("东吴秀享AI超额收益系统")

    # ========== 侧边栏参数设置 ========== #
    with st.sidebar:
        st.header("参数设置")

        # 数据设置分组
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)

        # 模型设置分组
        with st.expander("模型设置", expanded=True):
            classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            if classifier_name == "深度学习":
                classifier_name = "MLP"  # 原逻辑：深度学习 -> MLP
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_method = st.selectbox(
                "类别不均衡处理", 
                ["过采样", "类别权重", 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek']
            )
            # 将“过采样”映射为“SMOTE”、将“类别权重”映射为“Class Weights”
            if oversample_method == "过采样":
                oversample_method = "SMOTE"
            if oversample_method == '类别权重':
                oversample_method = "Class Weights"

            # 是否使用“10×10 笛卡尔积”组合最佳模型 的开关
            use_best_combo = st.checkbox(
                "精细化筛选模型", 
                value=False, 
                help="若开启，则将训练10个峰模型与10个谷模型的所有组合(100)进行回测，选出收益最高的组合。"
            )

        # 特征设置分组
        with st.expander("特征设置", expanded=True):
            auto_feature = st.checkbox("自动特征选择", True)
            n_features_selected = st.number_input(
                "选择特征数量", 
                min_value=5, max_value=100, value=20, 
                disabled=auto_feature
            )

        
    # ========== 三个选项卡 ========== #
    tab1, tab2, tab3 = st.tabs(["训练模型", "预测", "上传模型预测"])

    # =======================================
    #         Tab1: 训练模型
    # =======================================
    with tab1:
        st.subheader("训练参数")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("训练开始日期", datetime(2000, 1, 1))
        with col2:
            train_end = st.date_input("训练结束日期", datetime(2020, 12, 31))

        # --- 点击“开始训练”才执行 train_model() ---
        if st.button("开始训练"):
            try:
                with st.spinner("数据预处理中..."):
                    # 读取相同区间的数据
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))

                    # 做标注
                    df_preprocessed, all_features = preprocess_data(
                        df, 
                        N, 
                        mixture_depth, 
                        mark_labels=True
                    )

                with st.spinner("模型训练中..."):
                    # 正式调用 train_model()
                    (peak_model, peak_scaler, peak_selector,
                     peak_selected_features, all_features_peak, peak_best_score,
                     peak_metrics, peak_threshold,
                     trough_model, trough_scaler, trough_selector,
                     trough_selected_features, all_features_trough,
                     trough_best_score, trough_metrics, trough_threshold) = train_model(
                        df_preprocessed,
                        N,
                        all_features,
                        classifier_name,
                        mixture_depth,
                        n_features_selected if not auto_feature else 'auto',
                        oversample_method
                    )

                    # 保存模型到 session_state
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

                # 训练完成后，再画一次波峰波谷图
                peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
                troughs = df_preprocessed[df_preprocessed['Trough'] == 1]
                fig = plot_candlestick(
                    df_preprocessed,
                    symbol_code,
                    train_start.strftime("%Y%m%d"),
                    train_end.strftime("%Y%m%d"),
                    peaks=peaks,
                    troughs=troughs
                )
                st.plotly_chart(fig, use_container_width=True, key="chart1")

            except Exception as e:
                st.error(f"训练失败: {str(e)}")

        # ---------- (A) 可视化训练集：仅做数据预处理(标记峰/谷) ----------
        try:
            st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
            symbol_type = 'index' if data_source == '指数' else 'stock'
            data = read_day_from_tushare(symbol_code, symbol_type)
            df = select_time(data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))

            df_preprocessed, all_features = preprocess_data(
                df, 
                N, 
                mixture_depth,
                mark_labels=True
            )

            peaks = df_preprocessed[df_preprocessed['Peak'] == 1]
            troughs = df_preprocessed[df_preprocessed['Trough'] == 1]

            fig = plot_candlestick(
                df_preprocessed,
                symbol_code,
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                peaks=peaks,
                troughs=troughs
            )
            st.plotly_chart(fig, use_container_width=True, key="chart2")

        except Exception as e:
            st.warning(f"可视化失败: {e}")

    # =======================================
    #    Tab2: 自动训练 + 预测 + 保存模型
    # =======================================
    with tab2:
        if not st.session_state.get('trained', False):
            st.warning("请先完成模型训练")
        else:
            st.subheader("预测参数")

            # 1) 预测起止日期
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2:
                pred_end = st.date_input("预测结束日期", datetime.now())

            # 2) 策略选择
            with st.expander("策略选择", expanded=False):
                load_custom_css()

                # 示例：追涨策略
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab2"
                    )

                # 示例：止损策略
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab2")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab2"
                    )

                # 示例：高点需创X日新高
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab2")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab2"
                    )

            if st.button("开始预测"):
                try:
                    # 读取 & 预处理预测数据
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    new_df = select_time(data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))

                    df_preprocessed, all_features = preprocess_data(
                        new_df,
                        st.session_state.models.get('N', 30),  # 从已训练模型中取 N
                        st.session_state.models.get('mixture_depth', 1),
                        mark_labels=True
                    )

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # (单峰+单谷) 或 (10×10 笛卡尔积)
                    best_excess = -np.inf
                    best_models = None
                    final_result = None
                    final_bt = {}
                    final_trades_df = pd.DataFrame()

                    enable_chase_val = enable_chase
                    enable_stop_loss_val = enable_stop_loss
                    enable_change_signal_val = enable_change_signal

                    # 为了避免冲突，重新获取当前策略参数
                    n_buy_val = st.session_state.get("n_buy_tab2", 10)
                    n_sell_val = st.session_state.get("n_sell_tab2", 10)
                    n_newhigh_val = st.session_state.get("n_newhigh_tab2", 60)

                    # ------------------ 不使用笛卡尔积 ------------------
                    if not use_best_combo:
                        # (单峰+单谷) 重复训练10次，选最佳
                        for i in range(10):
                            status_text.text(f"正在进行第 {i+1}/10 次模型训练...")
                            progress_bar.progress((i+1) / 10)

                            try:
                                (peak_model, peak_scaler, peak_selector,
                                 _, all_features_peak, _,
                                 _, peak_threshold,
                                 trough_model, trough_scaler, trough_selector,
                                 _, all_features_trough,
                                 _, _, trough_threshold) = train_model(
                                    df_preprocessed,
                                    st.session_state.models['N'],
                                    all_features,
                                    classifier_name,
                                    st.session_state.models['mixture_depth'],
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )

                                # 回测
                                _, bt_result, _ = predict_new_data(
                                    new_df,
                                    peak_model, peak_scaler, peak_selector, all_features_peak, peak_threshold,
                                    trough_model, trough_scaler, trough_selector, all_features_trough, trough_threshold,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=30,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
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

                        if best_models is None:
                            raise ValueError("所有训练尝试均失败，无法找到可用模型。")

                        # 用最佳模型做最终预测
                        status_text.text("使用最佳模型进行最终预测...")
                        final_result, final_bt, final_trades_df = predict_new_data(
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

                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=30,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # ------------------ 使用笛卡尔积(10×10) ------------------
                    else:
                        peak_models = []
                        trough_models = []

                        # 先各训练10次峰模型和谷模型
                        for i in range(10):
                            status_text.text(f"正在训练第 {i+1}/10 个峰+谷模型...")
                            progress_bar.progress((i+1) / 10)

                            try:
                                (peak_model, peak_scaler, peak_selector,
                                 _, all_features_peak, _,
                                 _, peak_threshold,
                                 trough_model, trough_scaler, trough_selector,
                                 _, all_features_trough,
                                 _, _, trough_threshold) = train_model(
                                    df_preprocessed,
                                    st.session_state.models['N'],
                                    all_features,
                                    classifier_name,
                                    st.session_state.models['mixture_depth'],
                                    n_features_selected if not auto_feature else 'auto',
                                    oversample_method,
                                    window_size=30
                                )
                                peak_models.append((
                                    peak_model, peak_scaler, peak_selector, 
                                    all_features_peak, peak_threshold
                                ))
                                trough_models.append((
                                    trough_model, trough_scaler, trough_selector, 
                                    all_features_trough, trough_threshold
                                ))
                            except Exception as e:
                                st.error(f"第 {i+1} 次训练失败: {str(e)}")
                                continue

                        # 笛卡尔积组合
                        model_combinations = list(product(peak_models, trough_models))
                        total_combos = len(model_combinations)  # 预期 10×10=100
                        for idx, (peak_model_data, trough_model_data) in enumerate(model_combinations):
                            combo_progress = (idx + 1) / total_combos
                            status_text.text(f"正在回测第 {idx+1}/{total_combos} 个组合...")
                            progress_bar.progress(combo_progress)

                            pm, ps, psel, paf, pth = peak_model_data
                            tm, ts, tsel, taf, tth = trough_model_data

                            try:
                                _, bt_result, _ = predict_new_data(
                                    new_df,
                                    pm, ps, psel, paf, pth,
                                    tm, ts, tsel, taf, tth,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=30,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
                                current_excess = bt_result.get('超额收益率', -np.inf)
                                if current_excess > best_excess:
                                    best_excess = current_excess
                                    best_models = {
                                        'peak_model': pm,
                                        'peak_scaler': ps,
                                        'peak_selector': psel,
                                        'all_features_peak': paf,
                                        'peak_threshold': pth,

                                        'trough_model': tm,
                                        'trough_scaler': ts,
                                        'trough_selector': tsel,
                                        'all_features_trough': taf,
                                        'trough_threshold': tth
                                    }
                            except:
                                # 忽略某些异常
                                continue

                        if best_models is None:
                            raise ValueError("所有组合均测试失败，无法完成预测。")

                        # 用最佳组合做最终预测
                        status_text.text("使用最佳组合进行最终预测...")
                        final_result, final_bt, final_trades_df = predict_new_data(
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

                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=30,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # --- 将“最佳模型”写入 session_state，可后续上传下载 ---
                    if best_models is not None:
                        st.session_state.best_models = best_models
                        # 去掉原先的 expander，直接给出一个“保存最佳模型”按钮
                        st.download_button(
                            label="保存最佳模型",
                            data=pickle.dumps(best_models),
                            file_name="best_models.pkl",
                            mime="application/octet-stream"
                        )

                    # ========== 回测结果展示 ========== #
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('"波段盈"累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',     final_bt.get('年化夏普比率', 0)),
                    ]

                    # 前三列
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # 后三列
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # K线图：标注预测的峰/谷
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]

                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True,
                        selected_classifiers=[classifier_name],
                        bt_result=final_bt
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart3")

                    # 表格展示
                    col_left, col_right = st.columns(2)

                    # 预测明细
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    # 交易记录
                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[[
                                    '买入日', '买入原因', '买入价', 
                                    '卖出日', '卖出原因', '卖出价', 
                                    '持仓日', '盈亏'
                                ]].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")

                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

    # =======================================
    #    Tab3: 上传模型文件，独立进行预测
    # =======================================
    with tab3:
        st.subheader("上传模型文件（.pkl）并预测")
        st.markdown("在此页面可以上传之前已保存的最佳模型文件，直接进行预测。")

        # 1) 上传模型
        uploaded_file = st.file_uploader("选择本地模型文件进行预测：", type=["pkl"])
        if uploaded_file is not None:
            with st.spinner("正在加载模型..."):
                best_models_loaded = pickle.load(uploaded_file)
                st.session_state.best_models = best_models_loaded
                # 一旦加载了模型，就视为已“trained”，可以进行预测
                st.session_state.trained = True
            st.success("已成功加载本地模型，可进行预测！")

        if not st.session_state.trained or (st.session_state.best_models is None):
            st.warning("请先上传模型文件，或前往其他页面训练并保存模型。")
        else:
            st.markdown("### 预测参数")
            # 与 Tab2 布局一致
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start_up = st.date_input("预测开始日期(上传模型Tab)", datetime(2021, 1, 1))
            with col_date2:
                pred_end_up = st.date_input("预测结束日期(上传模型Tab)", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()

                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab3")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab3"
                    )

                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab3")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab3"
                    )

                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab3")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab3"
                    )

            # 2) 点击预测
            if st.button("开始预测(上传模型Tab)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    data = read_day_from_tushare(symbol_code, symbol_type)
                    new_df = select_time(data, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

                    # 这里的 N、mixture_depth 可从 st.session_state.models 中获取，
                    # 或直接从 best_models 里看看是否存了 N, mixture_depth。如果没有就设置默认值
                    N_val = st.session_state.models.get('N', 30)
                    mixture_val = st.session_state.models.get('mixture_depth', 1)

                    df_preprocessed, all_features = preprocess_data(
                        new_df,
                        N_val,
                        mixture_val,
                        mark_labels=True
                    )

                    # 执行预测
                    final_result, final_bt, final_trades_df = predict_new_data(
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

                        N_val,
                        mixture_val,
                        window_size=30,
                        eval_mode=False,
                        N_buy=n_buy_up,
                        N_sell=n_sell_up,
                        N_newhigh=n_newhigh_up,
                        enable_chase=enable_chase_up,
                        enable_stop_loss=enable_stop_loss_up,
                        enable_change_signal=enable_change_signal_up,
                    )

                    st.success("预测完成！（使用已上传模型）")

                    # 回测结果展示
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('"波段盈"累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',     final_bt.get('年化夏普比率', 0)),
                    ]

                    # 前三列
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # 后三列
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # K线图：标注预测的峰/谷
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]

                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start_up.strftime("%Y%m%d"),
                        pred_end_up.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart_upload_tab")

                    # 显示预测明细 & 交易记录
                    col_left, col_right = st.columns(2)
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[[
                                    '买入日', '买入原因', '买入价', 
                                    '卖出日', '卖出原因', '卖出价', 
                                    '持仓日', '盈亏'
                                ]].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

if __name__ == "__main__":
    main_product()
