# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


from data import load_data, preprocess_x, split_data
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("data/train_x.csv")
    y = load_data("data/train_y.csv")

    preprocessed_x, preprocessed_y, patientunit = preprocess_x(x, y)
   

    train_x, test_x, train_y, test_y = split_data(preprocessed_x, preprocessed_y)
    
    

    model = Model(args)  # you can add arguments as needed
    model.fit(train_x, train_y)
    y_pred = model.predict_proba(test_x)[:,1]
    
    aurroc = roc_auc_score(test_y, y_pred)
    print('AUROC score:', aurroc)

    x = load_data("data/test_x.csv")

    processed_x_test, do_not_use, patientunit = preprocess_x(x, y)
    prediction_probs = model.predict_proba(processed_x_test)[:,1]
    
    df = pd.DataFrame({'patientunitstayid': patientunit.astype('Int32'), 'hospitaldischargestatus': prediction_probs})
    
    df.to_csv('predictions.csv', index=False)
    



if __name__ == "__main__":
    main()
