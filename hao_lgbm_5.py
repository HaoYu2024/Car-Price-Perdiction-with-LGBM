import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.base import clone
import re

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import *
from sklearn.preprocessing import *

from lightgbm import LGBMRegressor
from sklearn.metrics import *

pd.set_option('display.max_columns', None)
from IPython.display import clear_output
from tqdm import tqdm, trange
from tabulate import tabulate
import random
import time
import logging
from IPython.display import display
from IPython.display import display, HTML
from colorama import Fore
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.ensemble import *
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#train = pd.read_csv(train_file_path, nrows=1000) 


callbacks = [log_evaluation(period=300), early_stopping(stopping_rounds=200)]

sample_sub = pd.read_csv('sample_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
Original = pd.read_csv('used_cars.csv')

#print(f"Length of test: {len(test)}")
#print(f"Length of sample_sub: {len(sample_sub)}")

Original[['milage', 'price']] = Original[['milage', 'price']].map(
    lambda x: int(''.join(re.findall(r'\d+', x))))

train = pd.concat([train, Original], ignore_index=True)


def update(df):
    
    t = 100
    
    df['accident'] = df['accident'].map({
        'None reported': 'not_reported',
        'At least 1 accident or damage reported': 'reported'
    })
    df['transmission'] = df['transmission'].str.replace('/', '').str.replace('-', '')
    df['transmission'] = df['transmission'].str.replace(' ', '_')
    
    cat_c = ['id','brand','model','fuel_type','engine','transmission','ext_col','int_col','accident','clean_title']
    
    re_ = ['model','engine','transmission','ext_col','int_col']
    
    for col in re_:
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < t, col] = "noise"
        
    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
        
    return df

train  = update(train)
test   = update(test)


def feature(df):
    current_year = datetime.now().year

    df['Vehicle_Age'] = current_year - df['model_year']

    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age']

    def extract_horsepower(engine):
        try:
            return float(engine.split('HP')[0])
        except:
            return 0

    def extract_engine_size(engine):
        try:
            return float(engine.split(' ')[1].replace('L', ''))
        except:
            return 0

    df['Horsepower'] = df['engine'].apply(extract_horsepower)
    df['Engine_Size'] = df['engine'].apply(extract_engine_size)
    df['Power_to_Weight_Ratio'] = df['Horsepower'] / df['Engine_Size']

    luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

    df['Accident_Impact'] = df.apply(lambda x: 1 if x['accident'] == 1 and x['clean_title'] == 0 else 0, axis=1)
    
    return df

train = feature(train)
test = feature(test)

X = train.drop(['price'], axis=1)
y = train['price']
cat_features = X.select_dtypes(include=['category']).columns.tolist()

#X = X.dropna()
#y = y[X.index]
#test = test.dropna()

train_ids = X['id']
test_ids = test['id'] 

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {test.shape}")
print(f"Training columns: {X.columns}")
print(f"Test columns: {test.columns}")
print(f"Features during training: {X.columns}")
print(f"Features during prediction: {test.columns}")
print("Columns in X but not in test:", set(X.columns) - set(test.columns))
print("Columns in test but not in X:", set(test.columns) - set(X.columns))






SEED = 601
n_splits = 5






def Train_ML_Optuna(X, y, test, is_luxury=False, n_splits=5):
    # Modify file names based on whether it's for luxury cars
    param_file = 'best_lgbm_params_luxury.json' if is_luxury else 'best_lgbm_params_non_luxury.json'
    
    # Load best_params from a saved JSON file for luxury or non-luxury cars
    try:
        with open(param_file, 'r') as f:
            best_params = json.load(f)
            print(f"Loaded existing best parameters from {param_file}.")
    except FileNotFoundError:
        print(f"No existing parameters found for {'luxury' if is_luxury else 'non-luxury'} cars. Please ensure optimization is run first.")
        return  # Exit the function if best_params are not found
    
    # Ensure n_splits is an integer (convert it if it's not)
    n_splits = int(n_splits)

    # Proceed with KFold cross-validation and model training
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=601)
    rmse_scores = []
    test_preds = np.zeros((test.shape[0], n_splits))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the model using best_params
        model = LGBMRegressor(**best_params, n_estimators=1000, n_jobs=8)

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',callbacks=callbacks)

        val_predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        rmse_scores.append(rmse)

        test_preds[:, fold] = model.predict(test)

    avg_rmse = np.mean(rmse_scores)
    mean_test_preds = np.mean(test_preds, axis=1)

    header = f"\n{'Final Validation RMSE:':<25} {avg_rmse:.5f}\n"
    print(header)
    
    return mean_test_preds





def optimize_and_update_best_params(X, y,is_luxury=False, max_iterations=0, n_trials_per_iteration=10, target_rmse=30000):
    best_rmse = np.inf
    best_params = None
    

    param_file = 'best_lgbm_params_luxury.json' if is_luxury else 'best_lgbm_params_non_luxury.json'

    try:
        with open(param_file, 'r') as f:
            best_params = json.load(f)
            print(f"Loaded existing best parameters from {param_file}.")
    except FileNotFoundError:
        print(f"No existing parameters found for {'luxury' if is_luxury else 'non-luxury'}. Starting fresh.")








    # Start optimization loop
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}:")
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
                'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 0.1),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1.0),
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'random_state': SEED ,
                'early_stopping_rounds':200,
                'device_type':'GPU',
                'max_bin': 63
               
            }
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            rmse_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = LGBMRegressor(**params, n_estimators=1000, n_jobs=-1)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

                val_predictions = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)
        
        # Run Optuna optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials_per_iteration)

        # Get the best parameters and RMSE from the current iteration
        iteration_best_params = study.best_params
        iteration_best_rmse = study.best_value

        if iteration_best_rmse < best_rmse:
            best_rmse = iteration_best_rmse
            best_params = iteration_best_params

        # Stop early if target RMSE is reached
        if best_rmse < target_rmse:
            print(f"Target RMSE of {target_rmse} achieved: {best_rmse}")
            break

    # Save the best parameters
    with open(param_file, 'w') as f:
        json.dump(best_params, f)
    print(f"Best parameters saved to {param_file}")

    return best_params

luxury_train = train[train['Is_Luxury_Brand'] == 1]
non_luxury_train = train[train['Is_Luxury_Brand'] == 0]

luxury_test = test[test['Is_Luxury_Brand'] == 1]
non_luxury_test = test[test['Is_Luxury_Brand'] == 0]

X_luxury = luxury_train.drop(['price'], axis=1)
y_luxury = luxury_train['price']

X_non_luxury = non_luxury_train.drop(['price'], axis=1)
y_non_luxury = non_luxury_train['price']

# Run optimization for luxury cars
print("\n--- Optimizing for Luxury Cars ---")
best_params_luxury = optimize_and_update_best_params(X_luxury, y_luxury, is_luxury=True)

# Run optimization for non-luxury cars
print("\n--- Optimizing for Non-Luxury Cars ---")
best_params_non_luxury = optimize_and_update_best_params(X_non_luxury, y_non_luxury, is_luxury=False)



# Make predictions using the trained model (assuming you've already trained your models)
luxury_preds = Train_ML_Optuna(X_luxury, y_luxury, luxury_test, is_luxury=True, n_splits=5)
non_luxury_preds = Train_ML_Optuna(X_non_luxury, y_non_luxury, non_luxury_test, is_luxury=False, n_splits=5)

# Assign predictions back to the respective test sets
luxury_test['price'] = luxury_preds
non_luxury_test['price'] = non_luxury_preds




# After predictions for luxury and non-luxury cars are made

# Calculate R² score and other metrics for luxury cars
#luxury_r2 = r2_score(luxury_test['price'], luxury_preds)
#luxury_rmse = np.sqrt(mean_squared_error(luxury_test['price'], luxury_preds))
#luxury_mae = mean_absolute_error(luxury_test['price'], luxury_preds)

#print("\nLuxury Car Metrics:")
#print(f"R² Score: {luxury_r2:.4f}")
#print(f"RMSE: {luxury_rmse:.4f}")
#print(f"MAE: {luxury_mae:.4f}")

# Calculate R² score and other metrics for non-luxury cars
#non_luxury_r2 = r2_score(non_luxury_test['price'], non_luxury_preds)
#non_luxury_rmse = np.sqrt(mean_squared_error(non_luxury_test['price'], non_luxury_preds))
#non_luxury_mae = mean_absolute_error(non_luxury_test['price'], non_luxury_preds)

#print("\nNon-Luxury Car Metrics:")
#print(f"R² Score: {non_luxury_r2:.4f}")
#print(f"RMSE: {non_luxury_rmse:.4f}")
#print(f"MAE: {non_luxury_mae:.4f}")

# Scatter plot for Luxury cars: Actual vs Predicted
#plt.figure(figsize=(10, 6))
#plt.scatter(luxury_test['price'], luxury_preds, alpha=0.5, color='blue')
#plt.plot([luxury_test['price'].min(), luxury_test['price'].max()],
#         [luxury_test['price'].min(), luxury_test['price'].max()], color='red', lw=2)
#plt.title("Luxury Cars: Actual vs Predicted Prices")
#plt.xlabel("Actual Prices")
#plt.ylabel("Predicted Prices")
#plt.show()

# Scatter plot for Non-Luxury cars: Actual vs Predicted
#plt.figure(figsize=(10, 6))
#plt.scatter(non_luxury_test['price'], non_luxury_preds, alpha=0.5, color='green')
#plt.plot([non_luxury_test['price'].min(), non_luxury_test['price'].max()],
#         [non_luxury_test['price'].min(), non_luxury_test['price'].max()], color='red', lw=2)
#plt.title("Non-Luxury Cars: Actual vs Predicted Prices")
#plt.xlabel("Actual Prices")
#plt.ylabel("Predicted Prices")
#plt.show()

# Optional: Compare R² and RMSE of both models for more insight
#print("\nComparison of Luxury vs Non-Luxury Model Performance:")
#print(f"Luxury R² Score: {luxury_r2:.4f} | Non-Luxury R² Score: {non_luxury_r2:.4f}")
#print(f"Luxury RMSE: {luxury_rmse:.4f} | Non-Luxury RMSE: {non_luxury_rmse:.4f}")







# Combine predictions from luxury and non-luxury test sets
final_predictions = pd.concat([luxury_test, non_luxury_test])

# Drop rows where the 'id' is not unique
#final_predictions = final_predictions.drop_duplicates(subset='id', keep='first')

# Ensure 'final_predictions' has the correct number of rows (125690)
final_predictions = final_predictions.head(125690)

# Create the submission file with the 'id' and 'price' columns
submission = pd.DataFrame({
    'id': final_predictions['id'],  # Ensure 'id' matches the test set's ids
    'price': final_predictions['price']*1.05  # Predictions made by the model
})

# Save the submission as a CSV file with exactly 125690 rows
submission.to_csv("Submission_Two_Models_6.csv", index=False)

# Output the first few rows of the submission
print(submission.head())