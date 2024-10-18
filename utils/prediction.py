"""
Make a prediction model for global sales
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Preprocessing
def preprocess_data(data):

    # Fill NaN with mean
    data['critic_score'] = data['critic_score'].fillna(data['critic_score'].mean())
    data['user_score'] = data['user_score'].fillna(data['user_score'].mean())
    
    # Drop rows with missing values 
    data = data.dropna(subset=['platform', 'rating', 'genre'])
    
    # Use One-hot encode categorical features
    categorical_features = ['platform', 'rating', 'genre']
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    
    return data_encoded


# Split train test data
def split_data(data_encoded):
    X = data_encoded.drop(columns=['total_sales', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'name',])
    y = data_encoded['total_sales']
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train[['critic_score', 'user_score', 'year_of_release']] = scaler.fit_transform(X_train[['critic_score', 'user_score', 'year_of_release']])
    X_test[['critic_score', 'user_score', 'year_of_release']] = scaler.transform(X_test[['critic_score', 'user_score', 'year_of_release']])
    return X_train, X_test, scaler


# Train random forest model
def train_rf_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


# Train xgboost model
def train_xgb_model(X_train, y_train, n_estimators, learning_rate, max_depth):
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model


# Evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Training performance
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    # Testing performance
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return y_train_pred, y_test_pred, train_mae, train_r2, test_mae, test_r2



# Get feature importance
def feature_importance(model, X_train):
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    return importance_df


# Predict Sales
def predict_sales(model, scaler, new_data):
    new_data[['critic_score', 'user_score']] = scaler.transform(new_data[['critic_score', 'user_score']])
    predicted_sales = model.predict(new_data)
    print(f'Predicted Global Sales: {predicted_sales}')
