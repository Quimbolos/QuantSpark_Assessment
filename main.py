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


# %% 

# CLASSIFICATION NEURAL NETWORK

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
import itertools
torch.manual_seed(10)

X, y = load_data('Left')

# Dataset Class
class CaseStudyDataSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = X , y
    # Not dependent on index
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = torch.tensor(self.y.iloc[index])
        return (features, label)

    def __len__(self):
        return len(self.X)


# Neural Networks Model 
class NeuralNetwork(torch.nn.Module):
    def __init__(self, nn_config):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module("Input Layer", torch.nn.Linear(11, nn_config['hidden_layer_width']))  # Input layer
        self.layers.add_module("ReLU", torch.nn.ReLU())
        for i in range(nn_config['depth'] - 2):
            self.layers.add_module("Hidden Layer", torch.nn.Linear(nn_config['hidden_layer_width'], nn_config['hidden_layer_width']))  # Hidden Layer
            self.layers.add_module("Hidden ReLU", torch.nn.ReLU())
        self.layers.add_module("Output Layer", torch.nn.Linear(nn_config['hidden_layer_width'], 1))  # Output layer with 1 neuron
        self.layers.add_module("Sigmoid", torch.nn.Sigmoid())  # Add sigmoid activation function

    def forward(self, features):
        return self.layers(features)

def generate_nn_configs():
    '''
        Generates a list of dictionaries with different neural networks models

        Parameters
        ----------
        None 

        Returns
        -------
        config_dict_list: list
            A list of dictionaries containing the neural network parameters (Optimiser, lr, hidden_layer_width and depth)   
    '''

    # Parameters to change are: Optimiser, lr, hidden_layer_width and depth
    combinations_dict = {
        'Optimisers':['SGD', 'Adam', 'Adagrad'],
        'lr':[0.01, 0.001, 0.0001, 0.00001],
        'hidden_layer_width':[32, 64, 128, 256],
        'depth':[50,100,150]
    }

    config_dict_list = []
    # For every possible combination of the combinations_dict create a custom dictionary that is later stored in config_dict_list
    for iteration in itertools.product(*combinations_dict.values()):
        config_dict = {
            'optimiser': iteration[0],
            'lr': iteration[1],
            'hidden_layer_width': iteration[2],
            'depth': iteration[3]
        }
        config_dict_list.append(config_dict)

    return config_dict_list


# Train function (Includes a Loss function, Optimiser and Tensorboard visualization)
def train(train_dataloader, validation_dataloader, nn_config, epochs=20):
    """
        Trains the Neural Network Model built using nn_config and using the training DataLoader. Then, the model is evaluated using the validation_dataloader
        
        Parameters
        ----------
        train_dataloader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_dataloader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset
        
        nn_config: dict 
            A dictionary containing the neural network configuration(Optimiser/LearningRate/Width/Depth)   

        Returns
        -------
        model: __main__.NeuralNetwork
            A model from pythorch
        
        training_duration: float
            The time taken to train the model
        
        interference_latency: float
            The average time taken to make a prediction  
    """

    # Define the model
    model = NeuralNetwork(nn_config)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss() 

    # Set optimiser with lr from nn_config
    if nn_config['optimiser'] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=nn_config['lr'])
    elif nn_config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=nn_config['lr'])
    elif nn_config['optimiser'] == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=nn_config['lr'])

    # Initialise TensorBoard writer
    writer = SummaryWriter()

    # Start the training_duration timer
    timer_start = time.time()

    # Train the model
    for epoch in range(epochs):
        batch_idx = 0
        current_loss = 0.0
        true_labels = []
        predicted_labels = []
        for batch in train_dataloader:
            features, labels = batch
            features = features.to(torch.float32) # Convert torch into the right format
            labels = labels.to(torch.float32) # Convert torch into the right format
            prediction = model(features)
            loss = loss_fn(prediction,labels)
            loss.backward() 
            optimiser.step() 
            optimiser.zero_grad() 
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            true_labels.extend(labels.tolist())
            predicted_labels.extend(torch.argmax(prediction, dim=1).tolist())
            print('Loss', ls)

        # Write the cumulative training loss for each batch
        writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        # writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        print("Loss avg", current_loss / batch_idx)

        # evaluate the model on the validation set for each batch
        batch_idx = 0
        current_loss = 0.0
        true_labels_val = []
        predicted_labels_val = []
        prediction_time_list = []
        for features, labels in validation_dataloader:
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            timer_start_ = time.time() # Start timer for interference_latency
            yhat = model(features)
            timer_end_ = time.time() # End timer for interference_latency
            batch_prediction_time = (timer_end_-timer_start_)/len(features) # Calculate interference_latency for each batch
            prediction_time_list.append(batch_prediction_time) # Store interference_latency for each batch
            loss = loss_fn(yhat,labels)
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            true_labels_val.extend(labels.tolist())
            predicted_labels_val.extend(torch.argmax(yhat, dim=1).tolist())
                

        # writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)
        writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)


    # End the training_duration timer
    timer_end = time.time()

    # Calculate training_duration timer
    training_duration = timer_end - timer_start

    # Calculate interference_latency
    interference_latency =  sum(prediction_time_list) / len(prediction_time_list)

    return model, training_duration, interference_latency

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss

def calculate_metrics(best_model, train_loader, validation_loader, test_loader):

    '''
        Calculates the f1 score, accuracy, precision, and recall for the train, validation, and testing datasets

        Parameters
        ----------
        
        best_model: __main__.NeuralNetwork
            A model from pytorch
        
        train_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset

        test_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the test dataset
        
        Returns
        -------
        train_metrics: dict
            A dictionary containing the metrics obtained using the training dataset

        validation_metrics: dict
            A dictionary containing the metrics obtained using the validation dataset

        test_metrics: dict
            A dictionary containing the metrics using the testing dataset
    '''
    
    # Calculate f1 score, accuracy, precision, and recall metrics

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the training dataset features
    for features, labels in train_loader:
        features = features.to(torch.float32) # Convert into the right format
        labels = labels.to(torch.long) # Convert into the right format
        prediction = best_model(features) 
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)

    train_f1score = f1_score(y, y_hat, average='weighted')
    train_accuracy = accuracy_score(y, y_hat)
    train_precision = precision_score(y, y_hat, average='weighted')
    train_recall = recall_score(y, y_hat, average='weighted')

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the validation dataset features
    for features, labels in validation_loader:
        features = features.to(torch.float32) # Convert into the right format
        labels = labels.to(torch.long) # Convert into the right format
        prediction = best_model(features)
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)

    validation_f1score = f1_score(y, y_hat, average='weighted')
    validation_accuracy = accuracy_score(y, y_hat)
    validation_precision = precision_score(y, y_hat, average='weighted')
    validation_recall = recall_score(y, y_hat, average='weighted')

    # Obtain predictions using the test dataset features
    for features, labels in test_loader:
        features = features.to(torch.float32) # Convert into the right format
        labels = labels.to(torch.long) # Convert into the right format
        prediction = best_model(features)
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)

    test_f1score = f1_score(y, y_hat, average='weighted')
    test_accuracy = accuracy_score(y, y_hat)
    test_precision = precision_score(y, y_hat, average='weighted')
    test_recall = recall_score(y, y_hat, average='weighted')

    train_metrics = {
        'F1 Score' : train_f1score,
        'Accuracy' : train_accuracy,
        'Precision': train_precision,
        'Recall': train_recall,
    }

    validation_metrics = {
        'F1 Score' : validation_f1score,
        'Accuracy' : validation_accuracy,
        'Precision': validation_precision,
        'Recall': validation_recall,
    }

    test_metrics = {
        'F1 Score' : test_f1score,
        'Accuracy' : test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
    }
    
    return train_metrics, validation_metrics, test_metrics

def find_best_nn(config_dict_list, train_loader, validation_loader, test_loader):
    '''
        Trains various Neural Network Models using the train function, calculates the metrics using the calculate_metrics and train functions, and finally returns the best model 

        Parameters
        ----------
        config_dict_list: list
            A list of dictionaries containing the neural network parameters (Optimiser, lr, hidden_layer_width and depth)   

        train_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset

        test_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the test dataset
        
        Returns
        -------
        None
    
    '''
    # For each configuration, redefine the nn_model and the training function
    for i, (nn_config) in enumerate(config_dict_list):

        best_metrics_ = None

        # Get the hyperparemeters
        best_hyperparameters = nn_config

        # Train the NN model using the model, the dataloaders and nn_config file
        best_model, training_duration, interference_latency = train(train_loader,validation_loader, nn_config)

        # Calculate the metrics
        train_metrics, validation_metrics, test_metrics = calculate_metrics(best_model, train_loader, validation_loader, test_loader)

        best_metrics = {

        'F1 Score' : [train_metrics.get('F1 Score'),validation_metrics.get('F1 Score'),test_metrics.get('F1 Score')],
        'Accuracy' : [train_metrics.get('Accuracy'),validation_metrics.get('Accuracy'),test_metrics.get('Accuracy')],
        'Precison' : [train_metrics.get('Precison'),validation_metrics.get('Precison'),test_metrics.get('Precison')],
        'Recall' : [train_metrics.get('Recall'),validation_metrics.get('Recall'),test_metrics.get('Recall')],
        'training_duration' : training_duration,
        'inference_latency' : interference_latency,
    }
        # Store the metrics, config, and model:
        if best_metrics_ == None or best_metrics.get('F1 Score')[1]>best_metrics_.get('F1 Score')[1]:
            best_model_ = best_model
            best_hyperparameters_ = best_hyperparameters
            best_metrics_ = best_metrics

        if i >= 40:
            break

    print(best_metrics_, best_hyperparameters_)

    return best_model_, best_hyperparameters_, best_metrics_


if __name__ == "__main__":

    # Define the DataSet
    dataset = CaseStudyDataSet()

    # Define the batch size
    batch_size = 30

    # Split the data 
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)   

    # Call the generte nn configs function to get the list of config dictionaries
    config_dict_list = generate_nn_configs()

    # Call the find best nn model function
    find_best_nn(config_dict_list, train_loader, validation_loader, test_loader)

# %%
