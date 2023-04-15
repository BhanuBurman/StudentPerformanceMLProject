import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into DataIngestion")
        try:
            df = pd.read_csv("notebook\data\student.csv")
            logging.info("Read data from the local database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header= True)
            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=12)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header= True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header= True)
            logging.info("Ingestion of the Data is cokmpleted")

            return {
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            }
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    ob = DataIngestion()
    ob.initiate_data_ingestion()