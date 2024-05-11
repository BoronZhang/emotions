import sklearn
from sklearn.base import ClassifierMixin
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import random
import time
from typing import Callable, Union

class Trainer:
    """
    A class to train models on the WESAD dataset\\
    The dataset features should be first extracted and put it in the `X`
    """
    def __init__(self, models_params:dict[Callable, dict[str, list]], X:pd.DataFrame, y:pd.DataFrame, split:bool=True) -> None:
        """
        Parameters
        -----
        `model_params`: a dictionary of models\\
        Keys are the models, and values are the corresponding parameters
        `X`: Data Frame\\
        columns are the extracted features of windows\\
        rows are the values of each window
        `y`: Data Frame\\
        the label of each window
        """
        self.models = []
        for model in models_params.keys():
            self.models.append(GridSearchCV(model(), models_params[model], scoring='f1_macro', cv=5))
        if split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = X.copy(), pd.DataFrame([]), y.copy(), pd.DataFrame([])
        self._best_estimator:sklearn.base.BaseEstimator
        self._f1_avg = 0.0
    
    def fit(self, verbose:int=0) -> None:
        """
        fit all the passed models and find the best estimator
        """
        # fit the models
        for model in self.models:
            if verbose:
                print(end=f'Training {model.estimator}')
                start_time:float = time.time()
            model.fit(self.X_train, self.y_train)
            if verbose:
                print(f"\rTraining {model.estimator} completed in {time.time() - start_time} secs")

            
            if model.best_score_ > self._f1_avg:
                self._f1_avg:float = model.best_score_
                self._best_estimator:sklearn.base.BaseEstimator = model.best_estimator_
        

    def predict_all(self, verbose:int=0, X_test=None) -> dict[ClassifierMixin, np.ndarray]:
        if X_test == None:
            X_test = self.X_test
        preds = {}
        for model in self.models:
            if verbose:
                print(f'Predicting {model}')
            preds[model] = model.predict(X_test)
        
        self.predictions = preds
        return preds
    
    def report_all(self, verbose:int=0, X_test=None):
        if not hasattr(self, 'predictions'):
            self.predict_all(verbose=verbose, X_test=X_test)
        reports = {}
        for model in self.predictions:
            report = classification_report(y_true=self.y_test, y_pred=self.predictions[model], output_dict=True)
            reports[model] = report
        
        return reports
    

class WESADTrainer:
    def __init__(self, models_params:dict[Callable, dict[str, list]], sensor:str, base_folder:str='WESAD/', data_prefix='raw', subjects=[], classes=[1, 2, 3, 4]) -> None:
        """
        Parameters
        ----
        `base_folder`: string\\
        the base location of datasets
        `filenames`: list of strings\\
        the list of filenames for each subject\\
        files must be in this location: <`base_folder`>/<subject_id>/data/<filename>\\
        the default is 
        """
        self.path = base_folder + "%s/" + data_prefix + "_data/" + sensor + ".csv"
        self.models_params = models_params
        self.subjects = subjects
        self.classes = classes
        
        
    def predict_all(self, verbose=0):
        """
        Parameters:
        -----
        
        """
        total_res:dict[str, Union[float, dict[str, float]]] = {i: {'precision': 0, 'recall': 0, 'f1-score': 0} for i in ['1', '2', '3', '4', 'weighted avg', 'macro avg']}
        total_res['accuracy'] = 0.0
        # for testNo in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
        for testNo in self.subjects:
            test = f"S{testNo}"
            if verbose:print(f"Testing {test}")
            inputs:pd.DataFrame = self.load_data(test=test, verbose=verbose) # a pd.DataFrame
            
            self.fit(df=inputs, verbose=verbose) # apres, models are in self._best_estimator
            test_df = pd.read_csv(self.path % test)
            last_label = max(self.classes) + 1
            for label in test_df['label'].unique():
                if label not in self.classes:
                    test_df.loc[test_df['label'] == label, 'label'] = last_label

            report = self.report(test_df)
            self.X_test = test_df
            self.y_pred = self.prediction

            if verbose:print(f"Report of {test}: {report}")
            for key1 in report.keys():
                if isinstance(report[key1], float):
                    total_res[key1] += report[key1]
                else:
                    for key2 in report[key1].keys():
                        if key2 != 'support':
                            total_res[key1][key2] += report[key1][key2]
            
        for key1 in report.keys():
            if isinstance(report[key1], float):
                total_res[key1] /= len(self.subjects)
            else:
                for key2 in report[key1].keys():
                    if key2 != 'support':
                        total_res[key1][key2] /= len(self.subjects)
        return total_res
            
            


    def load_data(self, test, verbose:int=0):
        inputs:list[pd.DataFrame] = []
        for s_num in self.subjects:
            s = f"S{s_num}"
            if '.' in s and s != test: #subjects are in the folders
                continue
            df = pd.read_csv(self.path % s)
            last_label = max(self.classes) + 1
            for label in df['label'].unique():
                if label not in self.classes:
                    df.loc[df['label'] == label, 'label'] = last_label

            inputs.append(df)
        if verbose:
            print()
        return pd.concat(inputs)
        
        
    def fit(self, df, verbose:int=0):
        X = df[[col for col in df.columns if col != 'label']]
        y = df['label']
        trainer = Trainer(self.models_params, X, y, split=False)
        trainer.fit(verbose=max(verbose-1, 0))
        self._best_estimator = trainer._best_estimator
        self.X_train = trainer.X_train
        self.y_train = trainer.y_train
        
    def predict(self, X_test:pd.DataFrame, verbose:int=0):
        pred = self._best_estimator.predict(X_test)
        print(f"predicting df of size {X_test.shape}, made {pred.shape}")
        self.prediction = pred
        return pred
    
    def report(self, df_test:pd.DataFrame, verbose:int=0):
        X_test = df_test[[col for col in df_test.columns if col != 'label']]
        y_test = df_test['label']
        self.predict(X_test=X_test, verbose=verbose)
        report = classification_report(y_true=y_test, y_pred=self.prediction, output_dict=True)
        return report
    
    

