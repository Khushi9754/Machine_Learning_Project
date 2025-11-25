from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
import sys

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transform(self):
        try:
            num_features = ['reading_score','writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
        
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('Standardscaler',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("OneHotEncoder",OneHotEncoder()),
                    ("StandardScaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical colums: {cat_features}")
            logging.info(f"Numerical colums: {num_features}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
        
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transform()
            
            target_col = 'math_score'
            num_features = ['reading_score','writing_score']
            
            input_features_train = train_df.drop(columns=[target_col])
            target_feature_train = train_df[target_col]
            
            input_features_test = test_df.drop(columns=[target_col])
            target_feature_test = test_df[target_col]
            
            logging.info(f"Applying preprocessing obj on train_df and test_df")
            
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)
            
            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test)]
            
            logging.info(f"Saved preprocessing obj")
            
            save_object(
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                    )
            
        except Exception as e:
            
            raise CustomException(e,sys) 
            