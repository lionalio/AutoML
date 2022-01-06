from libs import *
from modeling import *

df = pd.read_csv('house_price.csv')
X, y = df.drop('medv', axis=1), df['medv']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
config = {
    'metric': 'r2',
    'task': 'regression',
    'time_budget': 300
}

automl_rgs = get_model(X_train, y_train, **config)

y_pred = automl_rgs.predict(X_test)

print(automl_rgs.model.estimator)
