import os
import sys
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        logging.info("Error occured while creating Pickle file")
        raise CustomException(e, sys)
    


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            # Train Model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_predict = model.predict(X_test)

            # Get R2 score for test data
            test_model_score = r2_score(y_test, y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        logging.info("Exception Occured in Model Evaluation")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error occured while load the file object")
        raise CustomException(e,sys)
    
