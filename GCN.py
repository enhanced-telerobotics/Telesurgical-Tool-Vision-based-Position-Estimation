import os
from datetime import datetime
import json
import numpy as np
import random
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from typing import Optional, List


class StaticGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 device: torch.device, 
                 batch_size: int = 32, 
                 l2_reg: float = 0.0,
                 lr: float = 0.001, 
                 random_seed: Optional[int] = None):
        super(StaticGCN, self).__init__()

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
        # Model utilities
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.losses = []

        self.params = {'model': 'GCN',
                       'input_dim': input_dim,
                       'hidden_dim': hidden_dim,
                       'output_dim': output_dim,
                       'device': str(device),
                       'lr': lr,
                       'batch_size': batch_size,
                       'l2_reg': l2_reg,
                       'random_seed': random_seed}
        
        self.to(device)
        
        # Set the random seed if it's provided
        if random_seed is not None:
            os.environ['PYTHONHASHSEED']=str(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Global mean pooling
        x = global_mean_pool(x, batch=data.batch)
        return x

    def train(self, X_train, edge_index, y_train, epochs, use_tqdm=True, save_loss=False):
        super().train()

        # Data preparation
        X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
        data_list = [Data(x=X_train[i], edge_index=edge_index, y=y_train[i].unsqueeze(0))
                     for i in range(X_train.shape[0])]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(range(epochs), desc="Training Progress")
        else:
            pbar = range(epochs)

        best_loss = float('inf')
        for epoch in pbar:
            for data in loader:
                self.optimizer.zero_grad()
                out = self(data)
                loss = self.criterion(out, data.y)
                loss.backward()
                self.optimizer.step()

            if use_tqdm:
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix({'loss': loss.item()})

            if save_loss:
                self.losses.append(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.state_dict(), 'models/best_model.pth')

        if use_tqdm:
            pbar.close()

        if save_loss:
            self.params['losses'] = self.losses

            # Get the current time and format it as a string
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # Save the parameters and losses with a timestamp in the filename
            with open(f'results/losses_{timestamp}.json', 'w') as f:
                json.dump(self.params, f)
            os.rename('models/best_model.pth', f'models/best_model_{timestamp}.pth')

    def predict(self, X_test, edge_index_test, y_test):
        super().train(False)
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        edge_index_test = torch.tensor(
            edge_index_test, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float).to(self.device)

        test_data_list = [Data(x=X_test[i], edge_index=edge_index_test, y=y_test[i])
                          for i in range(X_test.shape[0])]
        test_loader = DataLoader(test_data_list, batch_size=1)

        predictions = np.zeros((0, 3))
        with torch.no_grad():
            for data in test_loader:
                pred = self(data)
                predictions = np.vstack((predictions, pred.cpu().numpy()))
        return predictions
