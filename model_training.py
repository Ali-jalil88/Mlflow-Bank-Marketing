# step 3 :
import mlflow
import mlflow.sklearn

def train_model(x_train, y_train, model, model_name, params):

    # step 1 : set model params
    model.set_params(**params) # set model params from params dict 

    # step 2 : train model
    model.fit(x_train, y_train)

    # step 3 : save model 
    mlflow.sklearn.log_model(model, model_name)

    # step 4 : log model params
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    return model
