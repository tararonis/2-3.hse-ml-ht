from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from dataclasses import dataclass
import re
import numpy as np
import math
import pickle



app = FastAPI()


## Helper class to process and clean data
@dataclass
class ProcessItem:
    def remove_units(self: object, val: object):        
        #if we have empty value("") insert NaN for feature work with values
        try:
            return np.NaN if not str(val).split(' ')[0] else float(str(val).split(' ')[0])
        except:
            print(f'Problem with {val}')

    def process_units_torque(self: object, val: object):
        if isinstance(val, str) or not math.isnan(val):
            units = ['( )', 'rpm', 'nm@', '(kgm@ rpm)', 'kgm at', "@", 'nm', 'nm at', 'kgm', '/', 'at', '//', '(', ')']
            
            pattern = '|'.join(map(re.escape, units))
            
            # Заменить найденные значения на пустую строку
            return re.sub(pattern, '', val.lower())
        
    def create_max_torque(self: object, val: object):
        try:
            if val is not None and isinstance(val, str):
                elements_to_split = ['-', '~', '+-']
                max_torque = val.split(' ')[1]
                if any(el in max_torque for el in elements_to_split):
                    split_pattern = '|'.join(map(re.escape, elements_to_split))
                    max_torque = re.split(split_pattern, max_torque)[1]
                return np.nan if max_torque == "" else float(max_torque.replace(',', '')) 
        except Exception as e:
            print(f'problem wth {val} || {e}')
            return np.nan
        
    def process_torque(self: object, val: object):
        try:
            if val is not None and isinstance(val, str):           
                torque = val.split(' ')[0]
                torque = torque.split('(')[0]            
                return np.nan if torque == "" else float(torque)
        except Exception as e:
            print(f'problem wth {val} || {e}')
            return np.nan      
    

class Item(BaseModel):
    name: str
    year: int    
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str    
    seats: float


class Items(BaseModel):
    objects: List[Item]


model = pickle.load(open('models/ridge_model_final.pkl', 'rb'))

def predict(item: Item) -> List:
    try:
        processor = ProcessItem()

        data_dict = item.dict()
        df_item = pd.DataFrame([data_dict])

        # Preparing data for model
        ## Cleaning data
        df_item['mileage'] = df_item['mileage'].apply(processor.remove_units)
        df_item['engine'] = df_item['engine'].apply(processor.remove_units)
        df_item['max_power'] = df_item['max_power'].apply(processor.remove_units)
        df_item['torque'] = df_item['torque'].apply(processor.process_units_torque)
        df_item['max_torque'] = df_item['torque'].apply(processor.create_max_torque)
        df_item['torque'] = df_item['torque'].apply(processor.process_torque)
        df_item['engine'] = df_item['engine'].apply(lambda val: int(val))

        ## Adding categorial features
        df_item = pd.get_dummies(df_item, drop_first=True, columns=['seats'])
        df_item = pd.get_dummies(df_item, drop_first=True, columns=['fuel'])
        df_item = pd.get_dummies(df_item, drop_first=True, columns=['seller_type'])
        df_item = pd.get_dummies(df_item, drop_first=True, columns=['transmission'])
        df_item = pd.get_dummies(df_item, drop_first=True, columns=['owner'])

        ## Adding aditional features
        df_item["year_square"] = df_item["year"] ** 2
        brand = df_item['name'].str.split(' ', expand=True)[0]        
        df_item[brand] = True
        df_item.drop(['name'], axis=1, inplace=True)        
        df_item = pd.get_dummies(df_item)      

        # Added the full model of the data frame that is required for the model
        df_model_data = pd.read_csv('models/model_data.csv')

        df_final_item = df_item.combine_first(df_model_data)        
        df_final_item = df_final_item.sort_index(axis=1)

        # Download the StandardScaler model for Standartization
        with open('models/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)  
        
        df_final_item = (scaler.transform(df_final_item))        
        
        pr = model.predict(df_final_item)        
        return abs(int(pr)) 
    except Exception as e:
        print("#########")
        print(e)       


@app.post("/predict_item")
def predict_item(item: Item) -> float:  
    print(f"##### Predicted price of the {item.name}")
    return predict((item))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    predictions = []
    for i in items:
        predictions.append(predict(i))
    print(f"##### Predicted {len(items)} Items")
    return predictions

@app.get("/")
def read_root():
    return {"U are the": "Nice person"}