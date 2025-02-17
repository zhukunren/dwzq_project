
import numpy as np
from function import *
import pandas as pd

def build_trades_from_signals(df, signal_df):

    # 修改列名转换方式：先复制，再修改列名
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    
    signal_df = signal_df.copy()
    signal_df.columns = signal_df.columns.astype(str).str.lower()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if not isinstance(signal_df.index, pd.DatetimeIndex):
        signal_df.index = pd.to_datetime(signal_df.index)

    signal_df = signal_df.reindex(df.index).fillna('')

    trades = []
    holding = False
    entry_date = None
    entry_price = None
    dates = df.index.to_list()

    for i in range(len(dates) - 1):  
        today = dates[i]
        next_day = dates[i + 1]
        direction_signal = signal_df.loc[today, 'direction'] if 'direction' in signal_df.columns else ''

        if not holding:
            if direction_signal == 'buy':
                # 下一交易日开盘买入
                holding = True
                entry_date = next_day
                entry_price = df.loc[next_day, 'open']
        else:
            if direction_signal == 'sell':
                # 下一交易日开盘卖出
                holding = False
                exit_date = next_day
                exit_price = df.loc[next_day, 'open']
                trade_return = exit_price / entry_price - 1 if entry_price else None
                hold_days = (df.index.get_loc(exit_date) - df.index.get_loc(entry_date))
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'hold_days': hold_days,
                    'return': trade_return
                })
                entry_date = None
                entry_price = None
    # 在for循环结束后，检查是否还在持仓
    if holding:
        holding = False
        exit_date = dates[-1]  # 最后一天
        exit_price = df.loc[exit_date, 'close']  # 或者 open, 或者您想要的价格
        trade_return = exit_price / entry_price - 1 if entry_price else None
        hold_days = (df.index.get_loc(exit_date) - df.index.get_loc(entry_date))
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'hold_days': hold_days,
            'return': trade_return
        })
        entry_date = None
        entry_price = None

    #print(pd.DataFrame(trades))
    return pd.DataFrame(trades)


def build_daily_equity_curve(df, trades_df, initial_capital=1_000_000):
    # 正确复制 DataFrame 并转换列名小写
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    
    trades_df = trades_df.copy()
    trades_df.columns = trades_df.columns.astype(str).str.lower()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    equity_curve = pd.DataFrame(index=df.index, columns=['equity'], data=np.nan)
    equity_curve.iloc[0, 0] = initial_capital

    # 标记每个交易日是否持仓
    position = pd.Series(data=0, index=df.index)  # 0=空仓,1=持仓
    for idx, row in trades_df.iterrows():
        ed = row['entry_date']
        xd = row['exit_date']
        if pd.isna(ed) or ed not in df.index:
            continue
        if pd.isna(xd) or xd not in df.index:
            # 如果exit_date为空或不在index里，就持仓到最后
            position.loc[ed:] = 1
        else:
            ed_loc = df.index.get_loc(ed)
            xd_loc = df.index.get_loc(xd)
            position.iloc[ed_loc:xd_loc] = 1

    # 计算每日涨跌幅
    df['daily_ret'] = df['close'].pct_change().fillna(0.0)

    # 逐日生成净值
    for i in range(1, len(df)):
        equity_curve.iloc[i, 0] = equity_curve.iloc[i-1, 0]
        if position.iloc[i-1] == 1:
            daily_ret = df['daily_ret'].iloc[i]
            equity_curve.iloc[i, 0] *= (1 + daily_ret)

    return equity_curve


def backtest_results(df, signal_df, initial_capital=1_000_000):
    """
    返回一个 dict，包含所有您想要的回测结果。
    
    参数
    -------
    df: 行情数据（index=交易日，含 'open','close' 列）
    signal_df: 含有 'direction' ('buy','sell') 列的信号
    initial_capital: 初始资金
    
    返回
    -------
    result: dict，包含 12 个字段，
      '同期标的涨跌幅'、
      '"波段盈"累计收益率'、
      '超额收益率'、
      '单笔交易最大收益'、
      '单笔交易最低收益'、
      '单笔平均收益率'、
      '收益率为正的交易笔数'、
      '收益率为负的交易笔数'、
      '持仓天数'、
      '空仓天数'、
      '交易笔数'、
      '胜率'
    """

    # ---------- (1) 构建交易表 ----------
    trades_df = build_trades_from_signals(df, signal_df)
    
    # ---------- (2) 构建日度净值 ----------
    equity_curve = build_daily_equity_curve(df, trades_df, initial_capital=initial_capital)

    # ---------- (3) 计算所有需要的指标 ----------

    # 3.1 同期标的涨跌幅
    # 以“首日的开盘”到“末日的收盘”来衡量
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower()
    signal_df = signal_df.copy()
    signal_df.columns = signal_df.columns.astype(str).str.lower()
    first_day = df.index[0]
    last_day = df.index[-1]
    benchmark_return = np.nan
    if 'open' in df.columns and 'close' in df.columns:
        start_price = df.loc[first_day, 'open']
        end_price   = df.loc[last_day, 'close']
        if start_price != 0:
            benchmark_return = end_price / start_price - 1
        else:
            benchmark_return = np.nan
    else:
        benchmark_return = np.nan

    # 3.2 策略累计收益率
    start_equity = equity_curve['equity'].iloc[0]
    end_equity   = equity_curve['equity'].iloc[-1]
    strategy_return = end_equity / start_equity - 1  # 波段盈

    # 3.3 超额收益
    excess_return = strategy_return - benchmark_return if (benchmark_return is not np.nan) else np.nan

    # 3.4 单笔交易相关
    if trades_df.empty:
        # 没有任何交易，则所有交易相关指标为空或0
        max_trade = None
        min_trade = None
        avg_trade = None
        pos_trades = 0
        neg_trades = 0
        num_trades = 0
        win_rate   = None
    else:
        max_trade = trades_df['return'].max()
        min_trade = trades_df['return'].min()
        avg_trade = trades_df['return'].mean()
        num_trades = len(trades_df)
        pos_trades = (trades_df['return'] > 0).sum()
        neg_trades = (trades_df['return'] < 0).sum()
        win_rate   = pos_trades / num_trades if num_trades > 0 else None

    # 3.5 持仓天数、空仓天数
    position = pd.Series(data=0, index=df.index)
    for idx, row in trades_df.iterrows():
        ed = row['entry_date']
        xd = row['exit_date']
        if pd.isna(ed) or ed not in df.index:
            continue
        if pd.isna(xd) or xd not in df.index:
            position.loc[ed:] = 1
        else:
            ed_loc = df.index.get_loc(ed)
            xd_loc = df.index.get_loc(xd)
            position.iloc[ed_loc:xd_loc] = 1

    holding_days = (position == 1).sum()
    noholding_days = (position == 0).sum()

    # 3.6 整理输出
    result = {
        '同期标的涨跌幅': benchmark_return,
        '"波段盈"累计收益率': strategy_return,  
        '超额收益率': excess_return,
        '单笔交易最大收益': max_trade,
        '单笔交易最低收益': min_trade,
        '单笔平均收益率': avg_trade,
        '收益率为正的交易笔数': pos_trades,
        '收益率为负的交易笔数': neg_trades,
        '持仓天数': holding_days,
        '空仓天数': noholding_days,
        '交易笔数': num_trades,
        '胜率': win_rate
    }

    return result
