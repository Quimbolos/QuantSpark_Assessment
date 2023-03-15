# %%

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from main import load_data

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

class ClassificationNN(nn.Module):
    def __init__(self, input_size, hidden_layer_width, output_size):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_width)
        self.fc2 = nn.Linear(hidden_layer_width, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, dataloader):
    model.eval()
    outputs = torch.tensor([], dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for inputs, labels in dataloader:
            predictions = model(inputs)
            outputs = torch.cat((outputs, predictions.argmax(dim=1)), dim=0)
            targets = torch.cat((targets, labels), dim=0)
    return outputs, targets

def calculate_metrics(outputs, targets):
    f1 = f1_score(targets, outputs, average='weighted')
    accuracy = accuracy_score(targets, outputs)
    precision = precision_score(targets, outputs, average='weighted')
    recall = recall_score(targets, outputs, average='weighted')
    return f1, accuracy, precision, recall

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
