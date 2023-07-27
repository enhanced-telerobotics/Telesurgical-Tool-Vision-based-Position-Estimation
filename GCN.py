import os
from datetime import datetime
import json
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Global mean pooling
        x = global_mean_pool(x, batch=data.batch)

        return x


# GraphModel for loading, training and predicting
class GraphModel:
    def __init__(self, X_train, edge_index, y_train, input_dim, hidden_dim, output_dim, batch_size, device):
        self.X_train = torch.tensor(X_train, dtype=torch.float).to(device)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        self.y_train = torch.tensor(y_train, dtype=torch.float).to(device)

        self.data_list = [Data(x=self.X_train[i], edge_index=self.edge_index,
                               y=self.y_train[i].unsqueeze(0)) for i in range(self.X_train.shape[0])]
        self.loader = DataLoader(
            self.data_list, batch_size=batch_size, shuffle=True)

        self.model = GCN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.losses = []

    def train(self, epochs, use_tqdm=True, save_loss=False):
        self.model.train()

        if use_tqdm:
            pbar = tqdm(range(epochs), desc="Training Progress")
        else:
            pbar = range(epochs)

        best_loss = float('inf')
        for epoch in pbar:
            for data in self.loader:
                self.optimizer.zero_grad()
                out = self.model(data)
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
                    torch.save(self.model.state_dict(), 'models/best_model.pth')

        if use_tqdm:
            pbar.close()

        if save_loss:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            with open(f'results/losses_{timestamp}.json', 'w') as f:
                json.dump({'losses': self.losses}, f)

            if os.path.exists('models/best_model.pth'):
                os.rename('models/best_model.pth', f'models/best_model_{timestamp}.pth')

    def predict(self, X_test, edge_index_test, y_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        edge_index_test = torch.tensor(
            edge_index_test, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float).to(self.device)

        test_data_list = [Data(x=X_test[i], edge_index=edge_index_test, y=y_test[i])
                          for i in range(X_test.shape[0])]
        test_loader = DataLoader(test_data_list, batch_size=1)

        predictions = []
        with torch.no_grad():
            for data in test_loader:
                pred = self.model(data)
                predictions.append(pred)
        return predictions
