from libs import *
from modeling import *
from sklearn.datasets import load_iris

dataset = load_iris()
X, y = dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
config = {
    'test_size': 0.2,
    'random_state': 123,
    'metric': 'accuracy',
    'task': 'classification',
    'time_budget': 300
}

automl_clf = get_model(X_train, y_train, **config)

y_pred = automl_clf.predict(X_test)

print(accuracy_score(y_test,y_pred))