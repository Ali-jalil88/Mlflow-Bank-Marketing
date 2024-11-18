'''
main file to run the mlflow pipeline
'''

# step 1 : import libraries
import mlflow
import mlflow.sklearn

from src.data_preperation import load_and_split_data
from src.models import get_model
from src.model_training import train_model
from src.model_evaluate import evaluate_model
from src.model_params import get_model_params

def main():

    # step 1 : set experiment name
    mlflow.set_experiment("iris-classification-7")

    # step 2 : load and split data
    x_train, x_test, y_train, y_test = load_and_split_data()
    
    # models 
    models = get_model()

    # params
    model_params = get_model_params()

    # train & evaluate each model
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # train model
            trained_model = train_model(x_train, y_train, model, model_name, model_params[model_name])

            # evaluate model
            evaluate_model(trained_model, x_test, y_test)
            

if __name__ == "__main__":
    main()

