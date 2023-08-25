import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_stacked_layers):

        super(GRU,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
    def predict(self, dataset):

        L = len(dataset)
        data_loader = DataLoader(dataset, batch_size=L, shuffle=False)
        for x, _ in data_loader:
            return self.forward(x).reshape(-1).tolist()
    
    def train(self, dataset, criterion = 'mse', optimizer = 'adam', lr = 0.01, batch_size = 1, epochs = 5, verbose = 5):
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Criterion
        if criterion == 'mse':
            criterion = nn.MSELoss()
        elif criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'mae':
            criterion = nn.L1Loss()

        # Optimizer
        if optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr = lr)


        losses = []
        for i in range(epochs):
            batch_n = 0
            for data in data_loader:
                X, y = data
                start_time = time.time()
                # Reset Gradients
                optimizer.zero_grad()

                # Forward Pass
                pred =  self.forward(X)

                # Loss Function
                loss = criterion(pred, y)

                loss.backward()

                # Upgrade Weights and Biases
                optimizer.step()

                batch_n += 1
                if verbose != 0 and batch_n % verbose == 0:
                    print('Epoch:', i + 1, '| Batch number:', batch_n , '| Loss:', loss.item(), '| Time:',   round(time.time() - start_time, 2), 's')


            losses.append(loss.item())

        return losses
        
