import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# Helper function for GridSearch
def looper(a, b, c):
    return list(range(a, b, c))


# Setting up dataset
melbourne_data = pd.read_csv('melb_data.csv')
melbourne_data.dropna(subset=['Price'], inplace=True)

# Preprocessing chosen features
features = ['Rooms', 'Bathroom', 'Landsize', 'Distance']
melbourne_copy = melbourne_data[features].copy()

# Handle missing values
imputer = SimpleImputer(strategy='median')
melbourne_copy = pd.DataFrame(imputer.fit_transform(melbourne_copy), columns=features)

# Splitting data
X = melbourne_copy
y = melbourne_data.Price
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.20)

# GridSearch for best Decision Tree parameters
param_grid = {
    'max_depth': looper(5, 31, 5),
    'min_samples_split': looper(2, 10, 2),
    'min_samples_leaf': looper(1, 20, 2)
}
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(train_X, train_y)
print("Best Parameters:", grid_search.best_params_)

# GridSearch for best Random Forest parameters
param_grid2 = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'n_estimators': looper(50, 250, 50)
}
grid_search2 = GridSearchCV(
    RandomForestRegressor(random_state=0),
    param_grid2,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)
grid_search2.fit(train_X, train_y)
print("Best Parameters:", grid_search2.best_params_)

# Decision Tree Model
melbourne_model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=9,
    random_state=0
)
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)


# Random Forest Model
model_rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=0)
model_rf.fit(train_X, train_y)
rf_predictions = model_rf.predict(val_X)


# Linear Regression Model
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)
model_lr = LinearRegression().fit(train_X, train_y)
lr_predictions = model_lr.predict(val_X)


# Feature Importance Plot 
feature_importance = melbourne_model.feature_importances_
sns.barplot(x=feature_importance, y=X.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree Model")
plt.show()

# Correlation Heatmap 
corr_matrix = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Distance', 'Price']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.xticks(rotation=45, ha='right')
plt.title("Feature Correlation Heatmap")
plt.show()

# Remove landsize == 0 and ensure len(y) for this part is straightened
mask = melbourne_copy['Landsize'] > 0
melbourne_copy = melbourne_copy[mask]
y = y[mask]  # Filter y accordingly

melbourne_copy['Price_per_m2'] = y / melbourne_copy['Landsize']
melbourne_copy['Price_per_m2'].fillna(melbourne_copy['Price_per_m2'].median(), inplace=True)

new_features = ['Rooms', 'Bathroom', 'Price_per_m2', 'Distance']
X = melbourne_copy[new_features]

# Training X and Y
train_X, val_X, train_y, val_y1 = train_test_split(X, y, random_state=0, test_size=0.20)
melbourne_model.fit(train_X, train_y)


results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Linear Regression", "New Features DT"],
    "MAE": [mean_absolute_error(val_y, val_predictions),
            mean_absolute_error(val_y, rf_predictions),
            mean_absolute_error(val_y, lr_predictions),
            mean_absolute_error(val_y1, melbourne_model.predict(val_X))]
})
print(results.sort_values(by="MAE"))
