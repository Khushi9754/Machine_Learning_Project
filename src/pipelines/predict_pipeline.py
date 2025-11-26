import sys 
import pandas as pd 
import os 

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    
    def __init__(self):
        pass
    
    #This func is going to do preprocessing and prediction
    #Features ---> data giving by user
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            print("Before Loading")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After loading")
            
            #Transform data
            data_scale = preprocessor.transform(features)
            
            pred = model.predict(data_scale)
            
            return pred
          
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        
        self.gender = gender
        
        self.race_ethnicity = race_ethnicity
        
        self.parental_level_of_education = parental_level_of_education
        
        self.lunch = lunch
        
        self.test_preparation_course = test_preparation_course
        
        self.reading_score = reading_score
        
        self.writing_score = writing_score
        
    def data_as_df(self):
        try:
            custom_data_input_dic = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            return pd.DataFrame(custom_data_input_dic)
        
        except Exception as e:
            raise CustomException(e,sys)