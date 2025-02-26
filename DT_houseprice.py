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


#This function is used in the GridSearchCV
def looper(a, b, c):
    return [i for i in range(a, b, c)]


# Load dataset
melbourne_data = pd.read_csv('melb_data.csv')

# Drop rows with missing target values (Price)
melbourne_data.dropna(subset=['Price'], inplace=True)


# Chosen features
features = ['Rooms', 'Bathroom', 'Landsize', 'Distance']


# Preprocessing data
melbourne_copy = melbourne_data[features].copy()


# Handle missing values
imputer = SimpleImputer(strategy='median')  #
melbourne_copy = pd.DataFrame(imputer.fit_transform(melbourne_copy), columns=features)


''' Describing the chosen features and splitting the data into test and training data'''
X = melbourne_copy
print(X.describe())
y = melbourne_data.Price
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.20)

#Grid searching for best parameters for decision tree
param_grid = {'max_depth': looper(1, 31, 2), 'min_samples_split': looper(2, 10, 1), 'min_samples_leaf': looper(1, 20, 1)}
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(train_X, train_y)
print("Best Parameters:", grid_search.best_params_)


#Grid searching for best parameters for random forest
param_grid2 = {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10], 'n_estimators': looper(50, 250, 50)}
grid_search2 = GridSearchCV(RandomForestRegressor(random_state=0), param_grid2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search2.fit(train_X, train_y)
print("Best Parameters:", grid_search2.best_params_)


# Model initialization and fitting
melbourne_model = DecisionTreeRegressor(max_depth=13, min_samples_split=2, min_samples_leaf=8, random_state=0)
melbourne_model.fit(train_X, train_y)


#Prediction
val_predictions = melbourne_model.predict(val_X)


# Evaluation metrics
mae = mean_absolute_error(val_y, val_predictions)
r2 = r2_score(val_y, val_predictions)

print(f"Decision Tree MAE: {mae:.2f}")

# Random Forest Regression
model_rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=0)
model_rf.fit(train_X, train_y)
rf_predictions = model_rf.predict(val_X)
print("Random Forest MAE:", mean_absolute_error(val_y, rf_predictions))


model_lr = LinearRegression().fit(train_X, train_y)
lr_predictions = model_lr.predict(val_X)
print("Linear Regression MAE:", mean_absolute_error(val_y, lr_predictions))


features_import = ['Rooms', 'Bathroom', 'Landsize', 'Distance', 'Price']
melbourne_copy_import = melbourne_data[features_import].copy()
feature_importance = melbourne_model.feature_importances_
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree Model")
plt.show()


corr_matrix = melbourne_copy_import.corr()


#Creating the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)


#Showing and labelling the heatmap
plt.title("Feature Correlation Heatmap")
plt.show()


# New features
X['Price_per_m2'] = y / X['Landsize']
X['Rooms_per_Bathroom'] = X['Rooms'] / (X['Bathroom'] + 1)

new_features = ['Rooms_per_Bathroom', 'Price_per_m2', 'Distance']


