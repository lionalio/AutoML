from libs import *

def get_model(X, y, **kwargs):
    automl_settings = {
        "time_budget": kwargs['time_budget'],  # in seconds
        "metric": kwargs['metric'],
        "task": kwargs['task'],
    }

    automl_clf = AutoML()
    automl_clf.fit(X, y, **automl_settings) 
    
    return automl_clf



