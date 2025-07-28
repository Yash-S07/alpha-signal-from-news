# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:04:02 2025

@author: Yash Singhal
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

#__________________________Configuration_______________________________________
ticker_name = 'OIL'
processed_data_path = f'{ticker_name}_processed_features.pkl'
n_splits = 5 # Number of folds to test on



def perform_walk_forward_validation(df):
    """
    Performs walk forward validation on the dataset
    Parameters
    ----------
    df : DataFrame
        The df with new features and embeddings.
    Returns
    -------
    None.
    """
    print(f"Starting Walk-Forward Validation with {n_splits} splits...")
    
    embedding_features = df.filter(like='emb_')
    engineered_features = df[['volatility', 'hour_of_day']] # Select your new features

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
        
        report = classification_report(Y_test,Y_pred,output_dict = True)
        fold_result.append(report)
        print(f"Fold {i+1} F1-Score (Macro Avg): {report['macro avg']['f1-score']:.2f}\n")
        
    #Aggregate and Display the Final Results:
    avg_precision_down = np.mean([res['-1']['precision'] for res in fold_result])
    avg_recall_down = np.mean([res['-1']['recall'] for res in fold_result])
    avg_f1_down = np.mean([res['-1']['f1-score'] for res in fold_result])
    
    
    avg_precision_up = np.mean([res['1']['precision'] for res in fold_result])
    avg_recall_up = np.mean([res['1']['recall'] for res in fold_result])
    avg_f1_up = np.mean([res['1']['f1-score'] for res in fold_result])
        
    
    
    avg_accuracy = np.mean([res['accuracy']for res in fold_result])
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print("\nAverage 'Down' Signal Performance:")
    print(f"  Precision: {avg_precision_down:.2f}")
    print(f"  Recall:    {avg_recall_down:.2f}")
    print(f"  F1-Score:  {avg_f1_down:.2f}")

    print("\nAverage 'Up' Signal Performance:")
    print(f"  Precision: {avg_precision_up:.2f}")
    print(f"  Recall:    {avg_recall_up:.2f}")
    print(f"  F1-Score:  {avg_f1_up:.2f}")
    

if __name__ == '__main__':
    df = pd.read_pickle(processed_data_path)
    
    if df.empty or df['label'].nunique() < 2:
        print("Not enough data or labels to train a model. Try scraping more pages or adjusting the PRICE_CHANGE_THRESHOLD.")
    else:
        perform_walk_forward_validation(df)
