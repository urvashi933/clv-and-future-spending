import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import gdown

# Create directories for outputs and dataset
os.makedirs('images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

dataset_path = 'dataset/part_5_customer_ltv_prediction.csv'

# Download dataset if not exists
if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    # Using the specific file ID from the provided folder
    file_id = '1BIrMxVmcLTXwowxkQ1HqMw8U1eu6dCE7' 
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, dataset_path, quiet=False)

# 1. Data Understanding & Loading
print("--- Loading Dataset ---")
df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")

# 2. Data Cleaning & Feature Engineering
print("\n--- Data Cleaning ---")
# Drop duplicates
df = df.drop_duplicates()

# Handle missing values
# Fill missing AnnualIncome with median
df['AnnualIncome'] = df['AnnualIncome'].fillna(df['AnnualIncome'].median())
# Fill missing AverageOrderValue with median
df['AverageOrderValue'] = df['AverageOrderValue'].fillna(df['AverageOrderValue'].median())

# Feature Engineering
# Total previous spending based on PreviousOrders and AverageOrderValue
df['TotalPreviousSpending'] = df['PreviousOrders'] * df['AverageOrderValue']

# Total Visits (Engagement)
df['TotalVisits'] = df['WebsiteVisits'] + df['AppSessions']

# Engagement Score (Arbitrary formula combining visits, sessions, and recency)
# Higher visits and lower days since purchase = higher score
df['EngagementScore'] = (df['TotalVisits'] * 10) / (df['DaysSinceLastPurchase'] + 1)

# Spending Growth: FutureSpending - Year1Spending (only for EDA/reference, but wait, FutureSpending is target)
# We can't use FutureSpending to engineer features for the model!
# We can just define feature 'AvgSpendingPerVisit'
df['AvgSpendingPerVisit'] = df['Year1Spending'] / (df['TotalVisits'] + 1)

print("Features engineered successfully.")

# 3. Exploratory Data Analysis
print("\n--- Exploratory Data Analysis ---")
sns.set_theme(style="whitegrid")

# Distribution of Future Spending
plt.figure(figsize=(8, 6))
sns.histplot(df['FutureSpending'], bins=30, kde=True, color='blue')
plt.title('Distribution of Future Spending')
plt.xlabel('Future Spending ($)')
plt.ylabel('Count of Customers')
plt.figtext(0.5, 0.01, "Interpretation: Most customers are in the lower spending bracket, with a few high-value outliers.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/future_spending_distribution.png')
plt.close()

# Relationship between income and spending
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AnnualIncome', y='FutureSpending', data=df, alpha=0.6)
plt.title('Annual Income vs Future Spending')
plt.xlabel('Annual Income ($)')
plt.ylabel('Future Spending ($)')
plt.figtext(0.5, 0.01, "Interpretation: Higher annual income generally shows a positive trend towards higher future spending.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/income_vs_spending.png')
plt.close()

# Relationship between website visits and spending
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TotalVisits', y='FutureSpending', data=df, alpha=0.6, color='purple')
plt.title('Total Visits vs Future Spending')
plt.xlabel('Total Visits (Web + App)')
plt.ylabel('Future Spending ($)')
plt.figtext(0.5, 0.01, "Interpretation: Higher engagement (visits) generally correlates with higher spending, identifying active users.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/visits_vs_spending.png')
plt.close()

# Relationship between order count and future spending
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PreviousOrders', y='FutureSpending', data=df, alpha=0.6, color='orange')
plt.title('Previous Orders vs Future Spending')
plt.xlabel('Number of Previous Orders')
plt.ylabel('Future Spending ($)')
plt.figtext(0.5, 0.01, "Interpretation: Frequent buyers (high order count) are significantly more likely to spend more in the future.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/orders_vs_spending.png')
plt.close()

# Relationship between previous spending and future spending
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Year1Spending', y='FutureSpending', data=df, alpha=0.6, color='green')
plt.title('Year 1 Spending vs Future Spending')
plt.xlabel('Year 1 Spending ($)')
plt.ylabel('Future Spending ($)')
plt.figtext(0.5, 0.01, "Interpretation: Strong positive correlation; high Year 1 spending heavily predicts high future spending.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/year1_vs_future.png')
plt.close()

# Outliers in customer value
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Year1Spending'], color='cyan')
plt.title('Outlier Analysis in Customer Value (Year 1 Spending)')
plt.xlabel('Year 1 Spending ($)')
plt.figtext(0.5, 0.01, "Interpretation: Significant outliers exist in the higher spending range, representing the 'whales' of the customer base.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.15)
plt.savefig('images/spending_outliers.png')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Customer Features')
plt.figtext(0.5, 0.01, "Interpretation: Future spending is highly correlated with Year 1 Spending and Average Order Value.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.1)
plt.savefig('images/correlation_heatmap.png')
plt.close()

# 4. Customer Segmentation
print("\n--- Customer Segmentation ---")

# Defining advanced segments using business rules
def segment_customers(row):
    # Quantile thresholds
    high_spending_threshold = df['Year1Spending'].quantile(0.75)
    low_spending_threshold = df['Year1Spending'].quantile(0.25)
    high_engagement_threshold = df['EngagementScore'].quantile(0.75)
    low_engagement_threshold = df['EngagementScore'].quantile(0.25)
    
    # 1. At-risk Valuable: High spenders who haven't purchased recently
    if row['Year1Spending'] >= high_spending_threshold and row['DaysSinceLastPurchase'] > 30:
        return 'At-risk Valuable'
    
    # 2. High-potential: Medium spenders with high income and high engagement
    if (low_spending_threshold < row['Year1Spending'] < high_spending_threshold) and \
       row['AnnualIncome'] > df['AnnualIncome'].median() and \
       row['EngagementScore'] > high_engagement_threshold:
        return 'High-Potential'
    
    # 3. Low-Engagement: Low visits and low engagement score
    if row['EngagementScore'] <= low_engagement_threshold:
        return 'Low-Engagement'
    
    # Standard Value-based segments for the rest
    if row['Year1Spending'] >= high_spending_threshold:
        return 'High-Value'
    elif row['Year1Spending'] >= low_spending_threshold:
        return 'Medium-Value'
    else:
        return 'Low-Value'

df['CustomerSegment'] = df.apply(segment_customers, axis=1)

# Save segmented distribution
plt.figure(figsize=(10, 6))
segment_order = ['High-Value', 'Medium-Value', 'Low-Value', 'High-Potential', 'Low-Engagement', 'At-risk Valuable']
sns.countplot(x='CustomerSegment', data=df, order=segment_order, palette='viridis')
plt.title('Advanced Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Number of Customers')
plt.figtext(0.5, 0.01, "Logic: Combined spending, engagement, and recency to identify specific growth and churn risks.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=45)
plt.savefig('images/customer_segments.png')
plt.close()

# 5. Future Spending Prediction (Modeling)
print("\n--- Modeling ---")
X = df.drop(columns=['CustomerID', 'FutureSpending', 'CustomerSegment'])
y = df['FutureSpending']

# Categorical & Numerical columns
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=[np.number]).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=5),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100)
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })
    
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df)

results_df.to_csv('outputs/model_evaluation.csv', index=False)

# Optional: Feature Importance for Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(random_state=42, n_estimators=100))])
rf_pipeline.fit(X_train, y_train)
feature_names = num_cols.tolist() + list(rf_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols))
importances = rf_pipeline.named_steps['model'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

print("\nPipeline execution complete. Artifacts saved in /images and /outputs.")
