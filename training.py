from sklearn.base import ClassifierMixin
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
import random

class Trainer:
    """
    A class to train models on the WESAD dataset\\
    The dataset features should be first extracted and put it in the `X`
    """
    def __init__(self, models_params:dict[ClassifierMixin, dict[str, list]], X:pd.DataFrame, y:pd.DataFrame) -> None:
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
        self.models:list[GridSearchCV] = []
        for model in models_params:
            self.models.append(GridSearchCV(model(), models_params[model], scoring='f1_macro', cv=5))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y)
        self._best_estimator = None
        self._f1_avg = 0.0
    
    def fit(self, verbose:int=0) -> None:
        """
        fit all the passed models and find the best estimator
        """
        # fit the models
        for model in self.models:
            if verbose:
                print(f'Training {model}')
            model.fit(self.X_train, self.y_train)

            
            if model.best_score_ > self._f1_avg:
                self._f1_avg, self._best_estimator = model.best_score_, model.best_estimator_
        

    def predict_all(self, verbose:int=0) -> dict[ClassifierMixin, np.array]:
        preds = {}
        for model in self.models:
            if verbose:
                print(f'Predicting {model}')
            preds[model] = model.predict(self.X_test)
        
        self.predictions = preds
        return preds
    
    def report_all(self, verbose:int=0):
        if not hasattr(self, 'predictions'):
            self.predict(verbose=verbose)
        reports = {}
        for model in self.predictions:
            report = classification_report(y_true=self.y_test, y_pred=self.predictions[model], output_dict=True)
            reports[model] = report
        
        return reports
    

class WESADTrainer:
    def __init__(self, models_params:dict[ClassifierMixin, dict[str, list]], base_folder='WESAD/', filenames:list[str]=None) -> None:
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
        if filenames == None:
            filenames = ['chest_ACC', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp', 'chest_Temp', 
                         'wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
        self.base_folder = base_folder
        self.filenames = filenames
        self.models_params = models_params
        self._best_estimator = {}
        
    def load_data(self, verbose:int=0):
        sensors = {sensor: [] for sensor in self.filenames}
        for s in os.listdir(self.base_folder):
            if '.' in s:
                continue
            for filename in self.filenames:
                sensors[filename].append(pd.read_csv(self.base_folder + s + '/data/' + filename + '.csv'))
            if verbose:print(end=f'\rSubject #{s} data loaded')
        if verbose:
            print()
        self.sensors_dfs = {sensor: pd.concat(sensors[sensor]) for sensor in sensors}
        if verbose:
            print("data frames concated successfully")

        
        
    def fit(self, verbose:int=0):
        if not hasattr(self, 'sensors_dfs'):
            raise RuntimeError("Before fitting data, first load it!")
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
        for sensor in self.sensors_dfs:
            df = self.sensors_dfs[sensor]
            X = df[['mean', 'std', 'min', 'max']]
            y = df['label']
            trainer = Trainer(self.models_params, X, y)
            trainer.fit(verbose=verbose-1)
            self._best_estimator[sensor] = trainer._best_estimator
            self.X_train.append(trainer.X_train)
            self.X_test.append(trainer.X_test)
            self.y_train.append(trainer.y_train)
            self.y_test.append(trainer.y_test)
            if verbose: print(f'Sensor {sensor} done!')
            del trainer
        if verbose:print()

        
    def predict(self, verbose:int=0):
        preds = {}
        final_pred = np.zeros(self.y_test.shape)
        for sensor in self._best_estimator:
            pred = self._best_estimator[sensor].predict(self.X_test)
            preds[sensor] = pred
        
        for i in range(len(preds)):
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for sensor in self._best_estimator:
                counts[preds[sensor][i]] += 1
            popular = max(counts.items(), key=lambda x: x[1])[1] # the most vote number
            items = [x[0] for x in counts.items() if x[1] == popular] # those with the most vote
            final_pred[i] = random.sample(items, 1)[0] # choose one of them randomly
        self.prediction = final_pred
        return final_pred
    
    def report(self, verbose:int=0):
        if not hasattr(self, 'prediction'):
            self.predict()
        report = classification_report(y_true=self.y_test, y_pred=self.prediction)
        return report





            

        
        



        


