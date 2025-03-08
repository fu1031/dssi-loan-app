"""
This module automates model training.
"""

import argparse
import pandas as pd
import numpy as np
import datetime
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from src import data_processor
from src import model_registry
from src.config import appconfig

logging.basicConfig(level=logging.INFO)

features = appconfig['Model']['features'].split(',')
categorical_features = appconfig['Model']['categorical_features'].split(',')
numerical_features = appconfig['Model']['numerical_features'].split(',')
label = appconfig['Model']['label']

def run(data_path, f1_criteria):
    """
    Main script to perform model training.
        Parameters:
            data_path (str): Directory containing the training dataset in csv
            f1_criteria (float): Minimum f1 score to achieve for training
        Returns:
            None: No returns required
    """
    logging.info('Process Data...')
    df = data_processor.run(data_path)
    
    numerical_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    # Train-Test Split
    logging.info('Start Train-Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(df[features], \
                                                        df[label], \
                                                        test_size=appconfig.getfloat('Model','test_size'), \
                                                        random_state=0)
    
    # Train Classifier
    logging.info('Start Training...')
    random_forest = RandomForestClassifier(n_estimators=appconfig.getint('Hyperparameters','rf_n_estimators'),
                                           max_depth=appconfig.getint('Hyperparameters','rf_max_depth'), 
                                           class_weight = appconfig.get('Hyperparameters','rf_class_weight'),
                                           n_jobs=appconfig.getint('Hyperparameters','rf_n_jobs'))
    
    clf = Pipeline(steps=[("preprocessor", preprocessor),\
                          ("binary_classifier", random_forest)
                         ])
    clf.fit(X_train, y_train)
    
    # Evaluate and Deploy
    logging.info('Evaluate...')
    score = f1_score(y_test, clf.predict(X_test), average='weighted')
    if score >= f1_criteria:
        logging.info('Deploy...')
        mdl_meta = { 'name': appconfig['Model']['name'], 'metrics': f"f1:{score}" }
        model_registry.register(clf, features, mdl_meta)
    
    logging.info('Training completed.')
    return None

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    argparser.add_argument("--f1_criteria", type=float)
    args = argparser.parse_args()
    run(args.data_path, args.f1_criteria)