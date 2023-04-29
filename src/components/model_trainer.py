# General Imports
import os
import sys
from dataclasses import dataclass

# Importing all the Regression techniques
from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# For Evaluating the models
from sklearn.metrics import r2_score

# Necesssary Import of the source
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("Split training testing data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Split training testing data completed...")
            models = {
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "K-NeiearestNeighbors":KNeighborsRegressor(),
                "XGBClassifier":XGBRegressor(),
                "CatBoostingClassifier":CatBoostRegressor(verbose=False),
                "AdaboostClassifier":AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(x_train=X_train,x_test=X_test, y_train=y_train,y_test = y_test,models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                                list(model_report.values()).index(best_model_score)
                            ]
            best_model = models[best_model_name]

            if best_model_score < 0.60 :
                raise CustomException("No Best Model Found")
            logging.info("Best Model Found on both training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
    





