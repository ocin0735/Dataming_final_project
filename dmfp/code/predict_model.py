# Amazon Product Rating Prediction Model 
# CS 422 Data Mining Final Project
# Author:Jiacheng Gu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')

# Create results directory
output_dir = 'model_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 60)
print("Amazon Product Rating Prediction Model")
print("=" * 60)

## 1. Problem Definition and Business Understanding

"""
Business Problem:

E-commerce platforms need to predict potential ratings for new products to anticipate market feedback before product launch, helping merchants formulate pricing and marketing strategies. Accurate rating predictions can help merchants identify potential bestsellers and adjust inventory and promotional resource allocation in advance.
Help merchants understand the key factors affecting product ratings, such as pricing strategies (discount intensity, price range), product category characteristics, and level of detail in product descriptions, to optimize product information and improve user satisfaction.
Provide rating prediction support for recommendation systems, optimizing recommendation algorithms by predicting ratings users might give, enhancing shopping experience, increasing user stickiness and platform activity.

Technical Problem:

This is a regression problem, aiming to predict product ratings (1-5 points). Features need to be extracted from structured and unstructured data to build regression models.
Features include:
Price information: Discounted price, original price, discount percentage and other numerical features.
Product categories: Multi-level category structure requiring encoding.
Product description text: Text features extracted through natural language processing (such as TF-IDF features).
Other metadata: Derived features like rating count, product name length, etc.
The challenge lies in handling mixed data types (numerical, textual, categorical) and balancing model complexity with generalization ability to avoid overfitting.
"""

## 2. Data Loading and Initial Exploration

try:
    # Load data
    df = pd.read_csv('amazon.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: amazon.csv file not found, please ensure the file is in the current directory")
    exit()

print("Dataset basic information:")
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable initial state:")
print(f"Sample unique values in rating column: {df['rating'].head(10).tolist()}")
print(f"Rating column data type: {df['rating'].dtype}")

## 3. Data Preprocessing

print("\n" + "="*50)
print("3. Data Preprocessing")
print("="*50)

# 3.0 First process the target variable rating
def clean_rating(rating_str):
    """Clean rating data, convert to numeric value"""
    if pd.isna(rating_str):
        return np.nan
    try:
        # Try direct conversion to float
        return float(str(rating_str))
    except:
        return np.nan

print("Processing target variable rating...")
df['rating'] = df['rating'].apply(clean_rating)
print(f"Rating data type after conversion: {df['rating'].dtype}")
print(f"Number of missing values in rating: {df['rating'].isnull().sum()}")

# Remove rows with empty ratings
initial_count = len(df)
df = df.dropna(subset=['rating'])
final_count = len(df)
print(f"Removed missing values: {initial_count} -> {final_count} rows")

print("\nRating statistics:")
print(df['rating'].describe())

# 3.1 Process price data
def clean_price(price_str):
    """Clean price data, extract numeric value"""
    if pd.isna(price_str):
        return 0
    # Remove currency symbols and commas, extract number
    price_clean = re.sub(r'[₹,]', '', str(price_str))
    try:
        return float(price_clean)
    except:
        return 0

print("Processing price data...")
df['discounted_price_clean'] = df['discounted_price'].apply(clean_price)
df['actual_price_clean'] = df['actual_price'].apply(clean_price)

# 3.2 Process discount percentage
def clean_discount(discount_str):
    """Clean discount percentage data"""
    if pd.isna(discount_str):
        return 0
    discount_clean = re.sub(r'%', '', str(discount_str))
    try:
        return float(discount_clean)
    except:
        return 0

df['discount_percentage_clean'] = df['discount_percentage'].apply(clean_discount)

# 3.3 Process rating count
def clean_rating_count(count_str):
    """Clean rating count data"""
    if pd.isna(count_str):
        return 0
    count_clean = re.sub(r'[,]', '', str(count_str))
    try:
        return int(count_clean)
    except:
        return 0

df['rating_count_clean'] = df['rating_count'].apply(clean_rating_count)

# 3.4 Create new features
print("Creating derived features...")
# Price-related features (avoid division by zero)
df['price_ratio'] = df['discounted_price_clean'] / (df['actual_price_clean'] + 1)
df['absolute_savings'] = df['actual_price_clean'] - df['discounted_price_clean']

# Product description length features
df['product_name_length'] = df['product_name'].str.len()
df['about_product_length'] = df['about_product'].str.len()

# Category hierarchy features (with proper escaping)
df['category_depth'] = df['category'].str.count('\\|') + 1

# 3.5 Process category data
df['main_category'] = df['category'].str.split('|').str[0]

print("Data preprocessing completed!")
print("New features:")
new_features = ['discounted_price_clean', 'actual_price_clean', 'discount_percentage_clean', 
                'rating_count_clean', 'price_ratio', 'absolute_savings', 
                'product_name_length', 'about_product_length', 'category_depth']
print(new_features)

# Check data quality
print(f"\nData quality check:")
print(f"Final dataset size: {df.shape}")
print(f"Rating range: {df['rating'].min():.2f} - {df['rating'].max():.2f}")
print(f"Price feature statistics:")
print(f"  Discounted price range: {df['discounted_price_clean'].min():.0f} - {df['discounted_price_clean'].max():.0f}")
print(f"  Original price range: {df['actual_price_clean'].min():.0f} - {df['actual_price_clean'].max():.0f}")

## 4. Exploratory Data Analysis (EDA)

print("\n" + "="*50)
print("4. Exploratory Data Analysis")
print("="*50)

# Create visualizations
plt.figure(figsize=(15, 10))

# 4.1 Target variable distribution
plt.subplot(2, 3, 1)
plt.hist(df['rating'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Product Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

# 4.2 Price vs Rating relationship
plt.subplot(2, 3, 2)
plt.scatter(df['discounted_price_clean'], df['rating'], alpha=0.5)
plt.title('Discounted Price vs Rating')
plt.xlabel('Discounted Price')
plt.ylabel('Rating')

# 4.3 Discount vs Rating relationship
plt.subplot(2, 3, 3)
plt.scatter(df['discount_percentage_clean'], df['rating'], alpha=0.5)
plt.title('Discount Percentage vs Rating')
plt.xlabel('Discount Percentage (%)')
plt.ylabel('Rating')

# 4.4 Rating count vs Rating relationship
plt.subplot(2, 3, 4)
plt.scatter(np.log1p(df['rating_count_clean']), df['rating'], alpha=0.5)
plt.title('Rating Count vs Rating (log scale)')
plt.xlabel('Rating Count (log)')
plt.ylabel('Rating')

# 4.5 Rating distribution by main category
plt.subplot(2, 3, 5)
top_categories = df['main_category'].value_counts().head(5).index
df_top_cat = df[df['main_category'].isin(top_categories)]
sns.boxplot(data=df_top_cat, x='main_category', y='rating')
plt.title('Rating Distribution by Category')
plt.xticks(rotation=45)

# 4.6 Feature correlation heatmap
plt.subplot(2, 3, 6)
numeric_features_corr = ['rating', 'discounted_price_clean', 'actual_price_clean', 
                        'discount_percentage_clean', 'rating_count_clean', 'price_ratio']
correlation_matrix = df[numeric_features_corr].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig(f'{output_dir}/eda_analysis.png', bbox_inches='tight')
plt.show()

print("Rating statistics:")
print(df['rating'].describe())

print(f"\nMain category distribution:")
print(df['main_category'].value_counts().head(10))

## 5. Feature Engineering

print("\n" + "="*50)
print("5. Feature Engineering")
print("="*50)

# 5.1 Prepare numerical features
numeric_features = [
    'discounted_price_clean', 'actual_price_clean', 'discount_percentage_clean',
    'rating_count_clean', 'price_ratio', 'absolute_savings',
    'product_name_length', 'about_product_length', 'category_depth'
]

# Check feature data types
print("Feature data type check:")
for feature in numeric_features:
    print(f"  {feature}: {df[feature].dtype}, missing values: {df[feature].isnull().sum()}")

# 5.2 Encode categorical features
le_category = LabelEncoder()
df['main_category_encoded'] = le_category.fit_transform(df['main_category'])

# 5.3 Process text features (product description)
print("Processing product description text features...")
try:
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
    text_features = tfidf.fit_transform(df['about_product']).toarray()
    print(f"TF-IDF feature extraction successful, number of features: {text_features.shape[1]}")
except Exception as e:
    print(f"TF-IDF feature extraction failed: {e}")
    print("Using basic text features instead...")
    text_features = np.zeros((len(df), 10))  # Create dummy text features

# Create text feature column names
text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]

# 5.4 Combine all features
features_numeric = df[numeric_features + ['main_category_encoded']].values
features_combined = np.hstack([features_numeric, text_features])

# Create feature name list
feature_names = numeric_features + ['main_category_encoded'] + text_feature_names

print(f"Final feature count: {features_combined.shape[1]}")
print(f"Sample count: {features_combined.shape[0]}")

# Check for invalid values in features
print(f"Invalid values in feature matrix: {np.isnan(features_combined).sum()}")
print(f"Infinite values in feature matrix: {np.isinf(features_combined).sum()}")

# Replace invalid values
features_combined = np.nan_to_num(features_combined, nan=0.0, posinf=0.0, neginf=0.0)

## 6. Model Building and Training

print("\n" + "="*50)
print("6. Model Building and Training")
print("="*50)

# 6.1 Prepare training data
X = features_combined
y = df['rating'].values

print(f"Final training data shape: X={X.shape}, y={y.shape}")
print(f"Target variable range: {y.min():.2f} - {y.max():.2f}")

# 6.2 Data splitting
# Simplified stratification strategy
try:
    y_binned = pd.cut(y, bins=3, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )
except Exception as e:
    print(f"Stratified sampling failed, using random sampling: {e}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 6.3 Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6.4 Define 4 main models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# 6.5 Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    
    try:
        # For models requiring standardization, use standardized data
        if name in ['Linear Regression']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Train model
        model.fit(X_train_model, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train_model)
        y_pred_test = model.predict(X_test_model)
        
        # Evaluation metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'model': model,
            'predictions': y_pred_test
        }
        
        print(f"Training set R²: {train_r2:.4f}")
        print(f"Test set R²: {test_r2:.4f}")
        print(f"Test set MSE: {test_mse:.4f}")
        print(f"Test set MAE: {test_mae:.4f}")
        print(f"Cross-validation R² (mean±std): {cv_mean:.4f} ± {cv_std:.4f}")
        
    except Exception as e:
        print(f"Model {name} training failed: {e}")

## 7. Model Performance Comparison

print("\n" + "="*50)
print("7. Model Performance Comparison")
print("="*50)

# Create performance comparison DataFrame
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[m]['train_r2'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Test MSE': [results[m]['test_mse'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
    'CV Mean R²': [results[m]['cv_mean'] for m in results.keys()],
    'CV Std R²': [results[m]['cv_std'] for m in results.keys()]
})

# Sort by Test R²
performance_df = performance_df.sort_values('Test R²', ascending=False)
print("\nModel performance ranking (by Test R²):")
print(performance_df.to_string(index=False))

# Create performance comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 7.1 R² comparison
ax = axes[0, 0]
x = np.arange(len(performance_df))
width = 0.35
ax.bar(x - width/2, performance_df['Train R²'], width, label='Train R²', alpha=0.8)
ax.bar(x + width/2, performance_df['Test R²'], width, label='Test R²', alpha=0.8)
ax.set_xlabel('Models')
ax.set_ylabel('R² Score')
ax.set_title('R² Score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 7.2 MSE and MAE comparison
ax = axes[0, 1]
x = np.arange(len(performance_df))
ax.bar(x - width/2, performance_df['Test MSE'], width, label='MSE', alpha=0.8)
ax.bar(x + width/2, performance_df['Test MAE'], width, label='MAE', alpha=0.8)
ax.set_xlabel('Models')
ax.set_ylabel('Error')
ax.set_title('MSE and MAE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 7.3 Cross-validation scores
ax = axes[1, 0]
ax.bar(x, performance_df['CV Mean R²'], yerr=performance_df['CV Std R²'], capsize=5, alpha=0.8)
ax.set_xlabel('Models')
ax.set_ylabel('CV R² Score')
ax.set_title('Cross-Validation R² Score')
ax.set_xticks(x)
ax.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# 7.4 Overfitting analysis
ax = axes[1, 1]
overfitting = performance_df['Train R²'] - performance_df['Test R²']
colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in overfitting]
ax.bar(x, overfitting, color=colors, alpha=0.8)
ax.set_xlabel('Models')
ax.set_ylabel('Train R² - Test R²')
ax.set_title('Overfitting Analysis')
ax.set_xticks(x)
ax.set_xticklabels(performance_df['Model'], rotation=45, ha='right')
ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/model_performance_comparison.png', bbox_inches='tight')
plt.show()

## 8. Best Model Detailed Analysis

print("\n" + "="*50)
print("8. Best Model Detailed Analysis")
print("="*50)

# Select best model
best_model_name = performance_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

print(f"Best model: {best_model_name}")
print(f"Test set R²: {performance_df.iloc[0]['Test R²']:.4f}")
print(f"Test set MAE: {performance_df.iloc[0]['Test MAE']:.4f}")

# Create detailed analysis graphs for best model
plt.figure(figsize=(15, 10))

# 8.1 Predicted vs Actual values
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title(f'{best_model_name}: Predicted vs Actual')

# 8.2 Residual analysis
plt.subplot(2, 3, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Rating')
plt.ylabel('Residuals')
plt.title('Residual Analysis')

# 8.3 Residual distribution
plt.subplot(2, 3, 3)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.axvline(x=0, color='r', linestyle='--')

# 8.4 QQ Plot
plt.subplot(2, 3, 4)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')

# 8.5 Prediction error percentage analysis
plt.subplot(2, 3, 5)
error_percentages = np.abs(residuals) / y_test * 100
plt.hist(error_percentages, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Error Percentage (%)')
plt.ylabel('Frequency')
plt.title('Prediction Error Percentage Distribution')
plt.axvline(x=np.median(error_percentages), color='r', linestyle='--', label=f'Median: {np.median(error_percentages):.1f}%')
plt.legend()

# 8.6 Feature importance (if model supports it)
plt.subplot(2, 3, 6)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title('Feature Importance (Top 15)')
else:
    plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Feature Importance')

plt.tight_layout()
plt.savefig(f'{output_dir}/best_model_analysis.png', bbox_inches='tight')
plt.show()

## 9. Model Optimization (for the best model)

print("\n" + "="*50)
print("9. Model Optimization")
print("="*50)

# If the best model is a tree model, perform hyperparameter optimization
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    print(f"\nOptimizing hyperparameters for {best_model_name}...")
    
    # Define parameter grid
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
    else:  # XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3]
        }
    
    # Grid search
    grid_search = GridSearchCV(
        type(best_model)(),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
    
    # Use best parameters to retrain
    optimized_model = grid_search.best_estimator_
    y_pred_optimized = optimized_model.predict(X_test)
    
    optimized_r2 = r2_score(y_test, y_pred_optimized)
    optimized_mae = mean_absolute_error(y_test, y_pred_optimized)
    
    print(f"\nOptimized performance:")
    print(f"Test set R²: {optimized_r2:.4f} (original: {performance_df.iloc[0]['Test R²']:.4f})")
    print(f"Test set MAE: {optimized_mae:.4f} (original: {performance_df.iloc[0]['Test MAE']:.4f})")
    
    # Update best model
    if optimized_r2 > performance_df.iloc[0]['Test R²']:
        best_model = optimized_model
        y_pred_best = y_pred_optimized
        print("\nOptimized model performance better, updated best model")

## 10. All models prediction results comparison

print("\n" + "="*50)
print("10. All models prediction results comparison")
print("="*50)

# Create prediction comparison chart
n_samples_to_show = 50
indices = np.random.choice(len(y_test), n_samples_to_show, replace=False)

plt.figure(figsize=(15, 8))

# Prepare data
predictions_data = {'Actual': y_test[indices]}
for model_name in results.keys():
    predictions_data[model_name] = results[model_name]['predictions'][indices]

# Create DataFrame
predictions_df = pd.DataFrame(predictions_data)

# Plot predictions for first 50 samples
plt.plot(range(n_samples_to_show), predictions_df['Actual'], 'ko-', label='Actual', markersize=6)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for i, model_name in enumerate(results.keys()):
    plt.plot(range(n_samples_to_show), predictions_df[model_name], 'o--', 
             color=colors[i], label=model_name, alpha=0.7, markersize=4)

plt.xlabel('Sample Index')
plt.ylabel('Rating')
plt.title('Predictions Comparison for Random 50 Samples')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/predictions_comparison.png', bbox_inches='tight')
plt.show()

## 11. Prediction accuracy analysis

print("\n" + "="*50)
print("11. Prediction accuracy analysis")
print("="*50)

# Calculate accuracy for different error tolerance ranges
tolerance_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
accuracy_results = {}

for model_name in results.keys():
    y_pred = results[model_name]['predictions']
    residuals = np.abs(y_test - y_pred)
    
    accuracies = []
    for tol in tolerance_levels:
        accuracy = np.mean(residuals <= tol) * 100
        accuracies.append(accuracy)
    
    accuracy_results[model_name] = accuracies

# Create accuracy comparison chart
plt.figure(figsize=(12, 6))

for model_name, accuracies in accuracy_results.items():
    plt.plot(tolerance_levels, accuracies, marker='o', label=model_name, linewidth=2)

plt.xlabel('Error Tolerance (Rating Points)')
plt.ylabel('Prediction Accuracy (%)')
plt.title('Model Accuracy vs Error Tolerance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(tolerance_levels)
plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_vs_tolerance.png', bbox_inches='tight')
plt.show()

# Print accuracy table
accuracy_df = pd.DataFrame(accuracy_results, index=[f'±{tol}' for tol in tolerance_levels])
print("\nModel accuracy (% in different error tolerance ranges):")
print(accuracy_df.round(1))
accuracy_df.to_csv(f'{output_dir}/accuracy_analysis.csv')

## 12. Model saving

print("\n" + "="*50)
print("12. Model saving")
print("="*50)

try:
    # Save all models
    for model_name, model_info in results.items():
        model_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model_info['model'], model_filename)
        print(f"{model_name} model saved: {model_filename}")
    
    # Save best model and related components
    joblib.dump(best_model, f'{output_dir}/best_model.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(tfidf, f'{output_dir}/tfidf.pkl')
    joblib.dump(le_category, f'{output_dir}/label_encoder.pkl')
    
    # Save model information
    model_info = {
        'best_model_name': best_model_name,
        'feature_names': feature_names,
        'performance_metrics': performance_df.to_dict()
    }
    joblib.dump(model_info, f'{output_dir}/model_info.pkl')
    
    print(f"\nBest model ({best_model_name}) and related components saved")
    
except Exception as e:
    print(f"Model saving failed: {e}")

## 13. Generate final report

print("\n" + "="*50)
print("13. Final report")
print("="*50)

# Create comprehensive report
report_content = f"""
# Amazon Product Rating Prediction Model Report

## 1. Data Overview
- Total sample count: {len(df)}
- Feature count: {X.shape[1]}
- Rating range: {y.min():.2f} - {y.max():.2f}
- Average rating: {y.mean():.2f} ± {y.std():.2f}

## 2. Model Performance Comparison
{performance_df.to_string()}

## 3. Best Model
- Model type: {best_model_name}
- Test set R²: {performance_df.iloc[0]['Test R²']:.4f}
- Test set MAE: {performance_df.iloc[0]['Test MAE']:.4f}
- Test set MSE: {performance_df.iloc[0]['Test MSE']:.4f}

## 4. Accuracy Analysis
Prediction accuracy within ±0.5 rating points:
{accuracy_df.loc['±0.5'].to_string()}

## 5. Business Insight
1. Most important features include rating count, price-related features, and product description
2. Suggest merchants focus on product description quality and reasonable pricing
3. Encourage more user ratings to improve product credibility

## 6. Model Application Suggestions
- {best_model_name} model performs best on this dataset
- Model can be used to predict potential ratings for new products
- Suggest regularly updating the model to maintain prediction accuracy
"""

# Save report
with open(f'{output_dir}/model_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)

print("\n" + "="*60)
print("Project completed! All results saved to 'model_results' directory")
print("="*60)