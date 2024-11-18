# step 5 :

def get_model_params():
    model_params = {
        "RandomForestClassifier": {"n_estimators": 300, "max_depth": 9},
        "LogisticRegression": {"C": 3.0, "penalty": "l2"},
        "SVC": {"C": 3.0, "kernel": "rbf"},
        "GaussianNB": {"var_smoothing": 1e-9},
        "KNeighborsClassifier": {"n_neighbors": 20},
        "DecisionTreeClassifier": {"max_depth": 9}
    }

    return model_params

