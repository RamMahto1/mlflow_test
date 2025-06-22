# experiments/decision_tree.py

import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.data_loader import load_data
import numpy as np
import pandas as pd


def run():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="DecisionTreeRegressor"):
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test,predictions))

        print("MSE:", mse)
        print("R2:", r2)
        print("MAE", mae)
        print("RMSE", rmse)
        

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("rmse",rmse)

        mlflow.sklearn.log_model(model, "decision_tree_model")
