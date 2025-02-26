import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

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

# Model initialization and fitting
melbourne_model = DecisionTreeRegressor(max_depth=12, min_samples_split=5, min_samples_leaf=7, random_state=0)
melbourne_model.fit(train_X, train_y)


#Prediction
val_predictions = melbourne_model.predict(val_X)


# Evaluation metrics
mae = mean_absolute_error(val_y, val_predictions)
r2 = r2_score(val_y, val_predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.3f}")


feature_importance = melbourne_model.feature_importances_
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree Model")
plt.show()

