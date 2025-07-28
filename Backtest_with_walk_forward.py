# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:49:55 2025

@author: Yash Singhal
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

#_____________________________Configuration____________________________________
ticker_name = 'OIL'
processed_data_path = f'{ticker_name}_processed_features.pkl'
n_splits = 5 # Number of folds to test on
transaction_cost = 0.0005 # 0.05% per trade(buy or sell)
risk_free_rate = 0 # Asssumed for the time being
look_forward_period = 4


def run_walk_forward_backtest(df):
    """
    Performs a robust walk forward backtest of the trading strategy
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    None.
    """
    print(f"Starting Walk-Forward Backtest with {n_splits} splits...")
    embedding_features = df.filter(like='emb_')
    engineered_features = df[['volatility', 'hour_of_day','Close']] # Select your new features

    X = pd.concat([embedding_features, engineered_features], axis=1)
    Y = df['label']
    
    #TimeSeriesSplit - Enusre we train on the past and test on future data
    tscv = TimeSeriesSplit(n_splits = n_splits)
    
    fold_result = []
    
    for i,(train_idx,test_idx) in enumerate(tscv.split(X)):
        print(f'-------------Processing fold - {i+1}/{n_splits}------------')
        
        
        #Create Training and testing data for current fold:
        X_train,X_test = X.iloc[train_idx],X.iloc[test_idx]
        Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
        
        print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')
        
        model = lgb.LGBMClassifier(
            objective = 'multiclass',
            num_class = 3,
            class_weight='balanced',
            random_state = 42
            )
        
        model.fit(X_train,Y_train)
        
        
        Y_pred = model.predict(X_test)
        
        fold_df = df.iloc[test_idx].copy()
        fold_df['predictions'] = Y_pred
        fold_df['Returns'] = fold_df['Close'].pct_change()#Calculating Returns
        fold_df['position'] = fold_df['predictions'] #The position I hold is the prediction
        
        #Return rn is because of the position i held previously 
        fold_df['strategy_return'] = fold_df['position'].shift(1)*fold_df['Returns']
        
        
        # Calculating transaction cost:
        # It is included only when position changes
        fold_df['trades'] = fold_df['position'].diff().abs()
        fold_df['TC'] = fold_df['trades'] * transaction_cost
        
        
        # Net Returns:
        fold_df['net_returns'] = fold_df['strategy_return'] - fold_df['TC']
        fold_result.append(fold_df['net_returns'])
        
    print("\n--- Final Backtest Results (Aggregated Across All Folds) ---")
    final_df = pd.concat(fold_result).fillna(0)
    
    final_df['cum_ret'] = (1+final_df['net_returns']).cumprod()
    
    periods_per_year = 252 * (24 / look_forward_period)
    mean_return = final_df['net_returns'].mean()
    std_dev = final_df['net_returnz'].std()
    
    # Avoid division by zero if there are no trades
    sharpe_ratio = ((mean_return - risk_free_rate) / std_dev) * np.sqrt(periods_per_year) if std_dev != 0 else 0
    
    # --- 5. REPORT FINAL METRICS ---
    total_return = final_df['cumulative_returns'].iloc[-1]
    
    print(f"Total Cumulative Return: {total_return:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    # Plot the final, robust equity curve
    final_df['cumulative_returns'].plot(
        title=f'Walk-Forward Backtest Equity Curve (Sharpe Ratio: {sharpe_ratio:.2f})',
        figsize=(12, 8),
        grid=True
    )
    plt.xlabel("Trade Number (across all folds)")
    plt.ylabel("Cumulative Returns")
    plt.show()
        
        
        
if __name__ == '__main__':
    df = pd.read_pickle(processed_data_path)
    if df.empty:
        print("Processed data file is empty. Please run the feature engineering script first.")
    else:
        run_walk_forward_backtest(df)
    