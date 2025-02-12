import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Data importing, selecting and modifying the selected data
melbourne_data = pd.read_csv('melb_data.csv')
melbourne_copy = melbourne_data.copy()
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Distance', 'Car']
melbourne_copy['Car'].fillna(0, inplace=True)
'''Fills the missing values in column 'Car' with 0 as no information given might be because there was none'''


''' Describing the chosen features and splitting the data into test and training data'''
X = melbourne_copy[melbourne_features]
print(X.describe())
y = melbourne_data.Price
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.20)


'''
output:
count  13580.000000  13580.000000   13580.000000  13580.000000  13580.000000
mean       2.937997      1.534242     558.416127     10.137776      1.602725
std        0.955748      0.691712    3990.669241      5.868725      0.966548
min        1.000000      0.000000       0.000000      0.000000      0.000000
25%        2.000000      1.000000     177.000000      6.100000      1.000000
50%        3.000000      1.000000     440.000000      9.200000      2.000000
75%        3.000000      2.000000     651.000000     13.000000      2.000000
max       10.000000      8.000000  433014.000000     48.100000     10.000000
'''


'''Modeling, fitting and prediction'''
melbourne_model = DecisionTreeRegressor(max_depth=12, min_samples_split=5, min_samples_leaf=7)
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# output: 275100.110895173
