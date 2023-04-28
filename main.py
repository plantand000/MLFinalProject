# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

from parser2 import parse
from data import load_data, preprocess_x, split_data
from model import Model


def main():
    args = parse()

    x = load_data("data/train_x.csv")
    y = load_data("data/train_y.csv")

    x, y = preprocess_x(x, y)

    train_x, train_y, test_x, test_y = split_data(x, y)

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    processed_x_train = preprocess_x(train_x)
    processed_x_test = preprocess_x(test_x)

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    model = Model(args)  # you can add arguments as needed
    model.fit(processed_x_train, train_y)
    x = load_data("test_x.csv")

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    processed_x_test = preprocess_x(x)

    prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results

    ########################


if __name__ == "__main__":
    main()
