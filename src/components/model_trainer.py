import os 
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and testing input data")
            
            X_train,X_test,y_train,y_test = (train_arr[:,:-1],
                                             test_arr[:,:-1],
                                            train_arr[:,-1],
                                            test_arr[:,-1])
            
            models = {
                "Linear Regression" : LinearRegression(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "RandomForest Regressor" : RandomForestRegressor(),
                "GradientBoosting Regressor" : GradientBoostingRegressor(),
                "KNeighbors Regressor" : KNeighborsRegressor(),
                "DecisionTree Regressor" : DecisionTreeRegressor(),
                "XGBRegressor" : XGBRFRegressor(),
                "catboost Regressor" : CatBoostRegressor(verbose = False)
            }
            
            para = {
                "Linear Regression" : {},
                
                "AdaBoost Regressor" : {
                    'n_estimators':[30,50,70,100,130,150],
                    'learning_rate':[0.01,0.1,0.3,0.5],
                    'loss':['exponential','linear','square']
                    },
                
                "RandomForest Regressor" : {
                    'criterion' : ['absolute_error','friedman_mse','poisson','squared_error'],
                    'n_estimators':[30,50,70,100,130,150]
                },
                
                "GraientBoosting Regressor" : {
                    'loss' : ['absolute_error','huber','quantile','squared_error'],
                    'learning_rate' : [0.01,0.1,0.3,0.5],
                    'n_estimators':[30,50,70,100,130,150]                          
                },
                
                "KNeighbours Regressor" : {
                    'n_neighbors' : [5,8,10,14,18],
                    'weights' : ['distance','uniform']
                },
                
                "DecisionTree Regressor" : {
                    'criterion' : ['absolute_error','friedman_mse','poisson','squared_error']
                },
                
                "XGBRegressor" : {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "CatBoost Regressor" : {
                    'learning_rate' : [0.01,0.3,0.5,0.1,0.7],
                    'iterations': [30, 50, 100],
                    'depth': [6,8,10]
                }
                
            }
            
            model_report :dict = evaluate_models(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                models=models,
                                                paras=para)
            
            #To get best model value
            best_model_score = max(model_report.values())
            
            #It return the key that has max value by using key=model_report.get
            best_model_name = max(model_report,key=model_report.get)
            
            best_model = models[best_model_name]
            
            if best_model_score <= 0.6 :
                raise CustomException("No Best model found")
            logging.info(f"Found Best model on both Training and Testing Data")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted_best_model = best_model.predict(X_test)
            
            rr2_score = r2_score(y_test,predicted_best_model)
            return rr2_score
            
        except Exception as e:
            raise CustomException(e,sys)