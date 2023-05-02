# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


from data import load_data, preprocess_x, split_data
from parser import parse
from model import Model


def main():
    args = parse()
    '''
    x = load_data("data/train_x.csv")
    y = load_data("data/train_y.csv")

    preprocessed_x, preprocessed_y, patientunit = preprocess_x(x, y)
   
    pd.DataFrame(preprocessed_x).to_csv('p_x.csv',index=False)
    pd.DataFrame(preprocessed_y).to_csv('p_y.csv',index=False)
    '''

    preprocessed_x = pd.read_csv('p_x.csv')
    preprocessed_y = pd.read_csv('p_y.csv')

    train_x, test_x, train_y, test_y = split_data(preprocessed_x, np.ravel(preprocessed_y))
    
    

    model = Model(args)  # you can add arguments as needed
    model.fit(train_x, train_y)
    y_pred,y_pred_acu = model.predict_proba(test_x)
    y_pred_acu =  [1 if x>=0.5 else 0 for x in y_pred_acu] 
    aurroc = roc_auc_score(test_y, y_pred[:,1])
    acu = accuracy_score(test_y,y_pred_acu)
    print('AUROC score:', aurroc)
    print('Acurracy score:', acu)

    '''

    x = load_data("data/test_x.csv")

    processed_x_test, do_not_use, patientunit = preprocess_x(x, y)
    prediction_probs = model.predict_proba(processed_x_test)[:,1]
    
    df = pd.DataFrame({'patientunitstayid': patientunit.astype('Int32'), 'hospitaldischargestatus': prediction_probs})
    
    df.to_csv('predictions.csv', index=False)
    '''



if __name__ == "__main__":
    main()
