import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        logging.info("Data Transformation Initiated")

        try:
            ## Define which columns should be ordinal-encoded, onehot encoded and which should be scaled
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 'distance']
            ordinal_cat_cols = ["Weather_conditions", "Road_traffic_density"]
            categorical_cols = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City']

            # Define Custom Ranking for Ordinal features
            Weather_conditions_categories = ["Stormy","Sandstorms","Windy","Fog","Cloudy","Sunny"]
            Road_traffic_density_categories = ["Jam","High", "Medium", "Low"]

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                
                ]
            )


            ## Ordinal Pipeline
            ord_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder(categories=[Weather_conditions_categories,Road_traffic_density_categories])),
                ("scaler", StandardScaler())


                ]
            )

            ## Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehotencoder",OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler())
                

                ]
            )



            ## Combine all 
            preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline,numerical_cols),
            ("ord_pipeline", ord_pipeline, ordinal_cat_cols),
            ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            logging.info("Pipeline Completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception occured at Data Transformation Stage")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe head : \n {train_df.head().to_string()}")
            logging.info(f"Test Dataframe head : ]\n {test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "Time_taken (min)"
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on trainig and testing datasets")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception Occured in initiate Data Transformation")
            raise CustomException(e,sys)