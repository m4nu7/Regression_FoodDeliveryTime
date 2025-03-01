import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Exception occured in preiction")
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Weather_conditions:object,
                 Road_traffic_density:object,
                 Vehicle_condition:int,
                 Type_of_order:object,
                 Type_of_vehicle:object,
                 multiple_deliveries:float,
                 Festival:object,
                 City:object,
                 distance:float):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival,
        self.City = City
        self.distance = distance


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age" : [self.Delivery_person_Age],
                "Delivery_person_Ratings" : [self.Delivery_person_Ratings],
                "Weather_conditions" : [self.Weather_conditions],
                "Road_traffic_density" : [self.Road_traffic_density],
                "Vehicle_condition" : [self.Vehicle_condition],
                "Type_of_order" : [self.Type_of_order],
                "Type_of_vehicle" : [self.Type_of_vehicle],
                "multiple_deliveries" : [self.multiple_deliveries],
                "Festival" : [self.Festival],
                "City" : [self.City],
                "distance" : [self.distance]

            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df

        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e,sys)
        
        

    