# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 01:51:50 2025

@author: Yash Singhal
"""


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#________________________Configuration_________________________________________
ticker_name = 'OIL'
processed_data_path = f'{ticker_name}_processed_features.pkl'

def train_and_evaluate(df):
    """
    Trains a lightgbm model and evaluate it's performance
    Parameters
    ----------
    df : DataFrame
        The aligned df from previous files with embeddings.
    Returns
    -------
    None.
    """
    print('Starting model training and evaluation...')
    
    embedding_features = df.filter(like='emb_')
    engineered_features = df[['volatility', 'hour_of_day']] # Select your new features

    X = pd.concat([embedding_features, engineered_features], axis=1)
    y = df['label']
    
    #Train_test_split But we do not shuffle the data
    #We train on the past data and test on future data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,shuffle = False)
    print(f'Train Size: {len(X_train)}, Test Size: {len(X_test)}')
    model = lgb.LGBMClassifier(objective = 'multiclass',num_class = 3,class_weight='balanced',random_state = 42)
    model.fit(X_train,Y_train)

    # Make predictions:
    Y_pred = model.predict(X_test)
    
    print('\n---------Classification Report----------')
    report = classification_report(Y_test,Y_pred,target_names = ['Down(-1)','Neutral(0)','Up(+1)'])
    print(report)
    
    
    print('\n---------Confusion Matrix--------------')
    cm = confusion_matrix(Y_test,Y_pred)
    sns.heatmap(cm,annot = True,fmt = 'd',xticklabels = ['Down','Neutral','Up'],yticklabels = ['Down','Neutral','Up'])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on Test Set')
    plt.show()
    
    print('\n---------Feature Importances----------')
    # Get feature names
    embedding_names = [f'emb_{i}' for i in range(768)]
    engineered_names = ['volatility', 'hour_of_day']
    feature_names = embedding_names + engineered_names
    
    # Create and display feature importance plot
    lgb.plot_importance(model, max_num_features=20, figsize=(10, 8), title='Top 20 Feature Importances')
    plt.show()


if __name__ == '__main__':
    df = pd.read_pickle(processed_data_path)
    
    
    if df.empty or df['label'].nunique() < 2:
        print("Not enough data or labels to train a model. Try scraping more pages or adjusting the PRICE_CHANGE_THRESHOLD.")
    else:
        train_and_evaluate(df)

    