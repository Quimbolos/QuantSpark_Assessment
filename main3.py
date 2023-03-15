
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_layer_width, depth, num_classes):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_width, hidden_layer_width) for i in range(depth-1)])
        self.fc2 = nn.Linear(hidden_layer_width, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = nn.functional.relu(hidden_layer(x))
        x = self.fc2(x)
        return x