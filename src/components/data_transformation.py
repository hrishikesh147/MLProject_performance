import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path=os.path.join("artifacts",'proprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            num_columns=['reading_score','writing_score']
            cat_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                ("simpleimputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            logging.info("numerical pipeline completed")

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one hot encoding",OneHotEncoder()),
                ("Standard scaler",StandardScaler(with_mean=False))   
                ]
            )

            logging.info("categorical pipeline completed")

            preprocessor=ColumnTransformer(
                [
                ("numerical columns",num_pipeline,num_columns),
                ("categorical pipeline",cat_pipeline,cat_columns)
                ]

            )

            logging.info("preprocessor returned completed")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading train and test data")

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            logging.info("separating input and target column for training data completed")

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("separating input and target column for testing data completed")

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("train arr and test arr completed")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info("saving file completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



            
        except Exception as e:
            raise CustomException(e,sys)






