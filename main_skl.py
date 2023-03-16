# %% SCIKIT LEARN CLASSIFICATION ML MODELS

import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def load_data(label):
    '''
    Returns the features and labels from the excel file in a tuple. It only includes numerical tabular data.

    Parameters
    ----------
    target: str
        A string with the pandas column to be used as the label

    Returns
    -------
    features, labels: tuple
        A tuple containing all the numerical features for the ML model and the target to predict (Price for the Night)
    '''
    
    current_directory = os.getcwd()
    csv_relative_directory = 'Data.csv'
    csv_directory = os.path.join(current_directory, csv_relative_directory)
    df = pd.read_csv(csv_directory)
    labels = df[label]
    features = df.drop(['Over18','StandardHours','complaintresolved','complaintyears',label], axis=1)

    # Preprocess the DataSet
    features['MonthlyIncome'] = features['MonthlyIncome'].str.capitalize()
    features['BusinessTravel'] = features['BusinessTravel'].str.replace('_',' ')
    labels = labels.map({'Yes': 1, 'No': 0})

    print(features.head())

    # Label Encoding for Gender / Income / Department / Business Travel 
    le_gender = preprocessing.LabelEncoder()
    le_gender.fit(['Female','Male'])
    features['Gender'] = le_gender.transform(features['Gender']) 

    le_Income = preprocessing.LabelEncoder()
    le_Income.fit([ 'Low', 'Medium', 'High'])
    features['MonthlyIncome'] = le_Income.transform(features['MonthlyIncome'])

    le_Department = preprocessing.LabelEncoder()
    le_Department.fit([ 'Research & Development', 'Sales', 'Human Resources'])
    features['Department'] = le_Department.transform(features['Department']) 

    le_BusinessTravel = preprocessing.LabelEncoder()
    le_BusinessTravel.fit([ 'Travel Rarely', 'Travel Frequently', 'Non-Travel'])
    features['BusinessTravel'] = le_BusinessTravel.transform(features['BusinessTravel']) 

    # # One Hot Encoding Age / Income / Department / Companies worked / Business Travel / Distance from Home / Job Satisfaction / Complaints / Salary Hike / Performance Rating / Total years working / Years at Company / Years Since Last Promotion
    
    # # Define columns to one-hot encode
    # columns_to_encode = ['Age', 'MonthlyIncome', 'Department','BusinessTravel', 'DistanceFromHome', 'complaintfiled','PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears','YearsAtCompany', 'YearsSinceLastPromotion','NumCompaniesWorked','JobSatisfaction']

    # # One-hot encode columns
    # features = pd.get_dummies(data=features, columns=columns_to_encode).astype(np.int64)

    print(features.head())

    return features, labels

def standardise_data(X):
    '''
    Standardises the data

    Parameters
    ----------
    X: pandas.core.frame.DataFrame
        A pandas DataFrame containing the features of the model

    Returns
    ----------
    X: pandas.core.frame.DataFrame
        A pandas DataFrame containing the standarised features of the model
    '''

    std = StandardScaler()
    scaled_features = std.fit_transform(X.values)
    X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

    return X

def split_data(X, y):
    '''
        Splits the data into training, validating and testing data

        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            A pandas DataFrame containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 

        Returns
        -------
        X_train, X_test: pandas.core.frame.DataFrame
            A set of pandas DataFrames containing the features of the model

        y_train, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels  
    '''

    np.random.seed(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    return X_train, X_test, y_train, y_test

def load_sci_kit_classification_models_and_hyperparameters():
    '''
        Returns the Sci-Kit Learn Classification models and a list of hyperparameters dictionaries for each model

        Parameters
        ----------
        None

        Returns
        -------
        models: list
            A list with instances for each sklearn model
        
        hyperparameters_dict: list
            A list containing hyperparameters dictionaries for each model
    '''

    models = [KNeighborsClassifier(), DecisionTreeRegressor(), SVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]

    hyperparameters_dict = [{ # KNeighborsClassifier

    'n_neighbors':[5,10,15,20,30],
    'weights':['uniform','distance'], 
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'leaf_size':[30], 
    'p':[2], 
    'metric':['minkowski'], 
    'metric_params':[None], 
    'n_jobs':[None]
},                      
                        { # DecisionTreeRegressor
    'criterion':['friedman_mse', 'squared_error','absolute_error'], 
    'splitter':['best','random'], 
    'max_depth':[None], 
    'min_samples_split':[2], 
    'min_samples_leaf':[1], 
    'min_weight_fraction_leaf':[0.0], 
    'max_features':['auto','sqrt',None], 
    'random_state':[None], 
    'max_leaf_nodes':[None], 
    'min_impurity_decrease':[0.0], 
    'ccp_alpha':[0.0]
},                     
                        { # SVC
    'C':[1.0], 
    'kernel':['rbf'], 
    'degree':[3], 
    'gamma':['scale'], 
    'coef0':[0.0], 
    'shrinking':[True], 
    'probability':[False], 
    'tol':[0.001], 
    'cache_size':[200], 
    'class_weight':[None], 
    'verbose':[False], 
    'max_iter':[-1], 
    'decision_function_shape':['ovr'], 
    'break_ties':[False], 
    'random_state':[None]
},
                        { # LogisticRegression
    'C': [1.0],
    'class_weight': ['balanced',None],
    'dual': [True, False],
    'fit_intercept': [True, False],
    'intercept_scaling': [1],
    'max_iter': [100],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'n_jobs': [None],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'random_state': [None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'tol': [0.0001],
    'verbose': [0],
    'warm_start': [True, False]
},
                        { # DecisionTreeClassifier
    'ccp_alpha': [0.0],
    'class_weight': [None],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [0.1, 1, 5, None],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'random_state': [None],
    'splitter': ['best', 'random']
},
                        { # RandomForestClassifier
    'bootstrap': [True, False],
    'ccp_alpha': [0.0],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None],
    'max_features': [1.0,'sqrt', 'log2', None],
    'max_leaf_nodes': [None],
    'max_samples': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'n_estimators': [50, 70, 100, 200],
    'n_jobs': [None],
    'oob_score': [True, False],
    'random_state': [None],
    'verbose': [0],
    'warm_start': [True, False]
},
                        { # GradientBoostingClassifier
    'ccp_alpha': [0.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'init': [None],
    'learning_rate': [0.1, 0.5, 1],
    'loss': ['log_loss', 'deviance', 'exponential'],
    'max_depth': [3],
    'max_features': ['auto', 'sqrt', 'log2',None],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'n_estimators': [10, 50, 70, 100],
    'n_iter_no_change': [None],
    'random_state': [None],
    'subsample': [1.0],
    'tol': [0.0001],
    'validation_fraction': [0.1],
    'verbose': [0],
    'warm_start': [True, False]
}]

    return models, hyperparameters_dict

def tune_classification_model_hyperparameters(model, X_train, X_test, y_train, y_test, hyperparameters_dict):
    '''
        Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

        Parameters
        ----------
        model: sklearn.model
            An instance of the sklearn model
        
        X_train, X_test: pandas.core.frame.DataFrame
            A set of pandas DataFrames containing the features of the model

        y_train, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels
        
        hyperparameters_dict: dict
            A dictionary containing a range of hyperparameters 

        Returns
        -------
        best_classification_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model         
    '''
    best_classification_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}
    
    X = X_train
    y = y_train
    
    model = model
    hyperparameters = hyperparameters_dict
    grid_search = GridSearchCV(model, hyperparameters, cv=5, scoring='f1')
    grid_search.fit(X, y)
    best_hyperparameters_dict[model] = grid_search.best_params_
    best_metrics_dict[model] = grid_search.best_score_
    if best_classification_model is None or best_metrics_dict[model] > best_metrics_dict[best_classification_model]:
        best_classification_model = model
        best_hyperparameters = best_hyperparameters_dict[model]
    
    model = best_classification_model.fit(X,y)
    best_classification_model = model
    y_pred_test = model.predict(X_test)

    best_metrics = {
        "F1 score" : f1_score(y_test, y_pred_test, average="macro"),
        "Precision":  precision_score(y_test, y_pred_test, average="macro"),
        "Recall" :  recall_score(y_test, y_pred_test, average="macro"),
        "Accuracy" :  accuracy_score(y_test, y_pred_test)
    }

    best_confusion_matrix = confusion_matrix(y_test, y_pred_test)
    df_cm = pd.DataFrame(best_confusion_matrix, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.show()

    return best_classification_model, best_hyperparameters, best_metrics, best_confusion_matrix


def find_best_classification_sci_kit_model(label, models,hyperparameters_dict):
    '''
        Imports and Standardizes the data, splits the dataset and finds the best-tuned classification model from the provided sklearn models and a range of its hyperparameters.       
        Finally, it returns the best classification model, its metrics and its hyperparameters
        
        Parameters 
        ----------
        label: str
            A string containing the label of the model

        models: list
            A list of models from sklearn 
        
        hyperparameters_dict: list
            A list of dictionaries containing a range of hyperparameters for each model

        Returns
        -------
        best_classification_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model 
    '''

    X, y = load_data('Left')
    # Import and standardize data
    X = standardise_data(X)

    # Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_classification_model_ = None
    best_hyperparameters_dict_ = {}
    best_metrics_dict_ = {}
    results_list = []

    # Tune models hyperparameters using GirdSearchCV
    for i in range(len(models)):
        print('MODEL:',i,'/',len(models))
        best_classification_model, best_hyperparameters_dict, best_metrics_dict, best_confusion_matrix = tune_classification_model_hyperparameters(models[i], X_train, X_test, y_train, y_test, hyperparameters_dict[i])
        results_list.append([best_classification_model, best_hyperparameters_dict, best_metrics_dict, best_confusion_matrix])
        if best_classification_model_ is None or (best_metrics_dict.get("F1 score").tolist()) > (best_metrics_dict_.get("F1 score").tolist()):
            best_classification_model_ = best_classification_model
            best_hyperparameters_dict_ = best_hyperparameters_dict
            best_metrics_dict_ = best_metrics_dict

    return best_classification_model_, best_hyperparameters_dict_, best_metrics_dict_, best_confusion_matrix, results_list

if __name__ == "__main__":

    # Sci-Kit Learn ML Classification Models
    models,hyperparameters_dict = load_sci_kit_classification_models_and_hyperparameters()
    best_regression_model, best_hyperparameters_dict, best_metrics_dict, best_confusion_matrix, results_list = find_best_classification_sci_kit_model('Left',models,hyperparameters_dict)


