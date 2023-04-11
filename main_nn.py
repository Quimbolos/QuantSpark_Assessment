# %%

import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# from torch.utils.tensorboard import SummaryWriter
import time
import itertools

torch.manual_seed(10)

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
    
    # Select Features & Labels
    labels = df[label]
    features = df.drop(['Over18','StandardHours','complaintresolved','complaintyears',label], axis=1)
    
    columns_to_encode = ['Gender','Department','BusinessTravel', 'complaintfiled','MonthlyIncome']
    columns_to_scale = ['Age', 'DistanceFromHome','PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears','YearsAtCompany', 'YearsSinceLastPromotion','NumCompaniesWorked','JobSatisfaction']

    scaler = StandardScaler()
    scaler.fit(df[columns_to_scale])

    # Preprocess the DataSet
    features['MonthlyIncome'] = features['MonthlyIncome'].str.capitalize()
    features['BusinessTravel'] = features['BusinessTravel'].str.replace('_',' ')
    labels = labels.map({'Yes': 1, 'No': 0})

    features_to_scale = df[columns_to_scale]
    features_to_encode = df[columns_to_encode]

    # # Label Encoding for Gender / Income / Department / Business Travel 
    # le_gender = preprocessing.LabelEncoder()
    # le_gender.fit(['Female','Male'])
    # features['Gender'] = le_gender.transform(features['Gender']) 

    # le_Income = preprocessing.LabelEncoder()
    # le_Income.fit([ 'Low', 'Medium', 'High'])
    # features['MonthlyIncome'] = le_Income.transform(features['MonthlyIncome'])

    # le_Department = preprocessing.LabelEncoder()
    # le_Department.fit([ 'Research & Development', 'Sales', 'Human Resources'])
    # features['Department'] = le_Department.transform(features['Department']) 

    # le_BusinessTravel = preprocessing.LabelEncoder()
    # le_BusinessTravel.fit([ 'Travel Rarely', 'Travel Frequently', 'Non-Travel'])
    # features['BusinessTravel'] = le_BusinessTravel.transform(features['BusinessTravel'])
    
    # le_gender = preprocessing.LabelEncoder()
    # le_gender.fit(['Female','Male'])
    # features['Gender'] = le_gender.transform(features['Gender']) 

    # One Hot Encoding Age / Income / Department / Companies worked / Business Travel / Distance from Home / Job Satisfaction / Complaints / Salary Hike / Performance Rating / Total years working / Years at Company / Years Since Last Promotion

    features_to_scale= scaler.transform(df[columns_to_scale])

    # One-hot encode columns
    features[columns_to_encode] = pd.get_dummies(data=features, columns=columns_to_encode).astype(np.int64)

    print(features.size)
    print(features.shape)
    print(labels.size)
    print(features.shape)

    return features, labels

X, y = load_data('Left')

# print(X.info())
# print(len(X))

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
        self.layers.add_module("Input Layer", torch.nn.Linear(212, nn_config['hidden_layer_width']))  # Input layer
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
def train(train_dataloader, validation_dataloader, nn_config, epochs=15):
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
    loss_fn = torch.nn.BCELoss() 

    # Set optimiser with lr from nn_config
    if nn_config['optimiser'] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=nn_config['lr'])
    elif nn_config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=nn_config['lr'])
    elif nn_config['optimiser'] == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=nn_config['lr'])

    # Initialise TensorBoard writer
    # writer = SummaryWriter()

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
            prediction=torch.squeeze(prediction)
            loss = loss_fn(prediction, labels)
            loss.backward() 
            optimiser.step() 
            optimiser.zero_grad() 
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            true_labels.extend(labels.tolist()) 
            predicted_labels.extend(prediction.tolist()) 
            # print('Loss', ls)

        # Write the cumulative training loss for each batch
        # writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        # writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        # print("Loss avg", current_loss / batch_idx)

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
            yhat=torch.squeeze(yhat)
            loss = loss_fn(yhat,labels)
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            true_labels_val.extend(labels.tolist())
            predicted_labels_val.extend(yhat.tolist())
                

        # writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)
        # writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)


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
        labels = labels.to(torch.float32) # Convert into the right format
        prediction = best_model(features) 
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)

  
    train_f1score = f1_score(y, y_hat)
    train_accuracy = accuracy_score(y, y_hat)
    train_precision = precision_score(y, y_hat)
    train_recall = recall_score(y, y_hat)

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the validation dataset features
    for features, labels in validation_loader:
        features = features.to(torch.float32) # Convert into the right format
        labels = labels.to(torch.float32) # Convert into the right
        prediction = best_model(features)
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)

    validation_f1score = f1_score(y, y_hat)
    validation_accuracy = accuracy_score(y, y_hat)
    validation_precision = precision_score(y, y_hat)
    validation_recall = recall_score(y, y_hat)

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the test dataset features
    for features, labels in test_loader:
        features = features.to(torch.float32) # Convert into the right format
        labels = labels.to(torch.float32) # Convert into the right format
        prediction = best_model(features)
        y.append(labels.detach().numpy())
        y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.argmax(y_hat, axis=1)


    print(np.sum(y), np.sum(y_hat))
    test_f1score = f1_score(y, y_hat)
    test_accuracy = accuracy_score(y, y_hat)
    test_precision = precision_score(y, y_hat)
    test_recall = recall_score(y, y_hat)

    # Print Test Confusion Matrix

    best_confusion_matrix = confusion_matrix(y_hat, y)
    df_cm = pd.DataFrame(best_confusion_matrix, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.show()

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

        if i >= 10:  # UNCOMMENT ON FOR TEST
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
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)   

    # Call the generte nn configs function to get the list of config dictionaries
    config_dict_list = generate_nn_configs()

    # Call the find best nn model function
    best_model_, best_hyperparameters_, best_metrics_ = find_best_nn(config_dict_list, train_loader, validation_loader, test_loader)

# %%
