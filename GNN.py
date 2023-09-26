import os
from datetime import datetime
import json
import numpy as np
import random
import torch
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from typing import Optional, List


class StaticGNN(torch.nn.Module):
    """
    A Graph Convolutional Network (GNN) model with static features.

    This class encapsulates the creation, training, and inference stages of a GNN model. The model
    uses Mean Squared Error (MSE) as the loss function and Adam as the optimizer. It is specifically
    designed for static features on nodes.

    Attributes:
        device: The device to run the model on (cpu or gpu).
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
        l2_reg: L2 regularization strength.
        weights: Weights for the output dimensions.
        params: A dictionary storing various hyperparameters and configurations.
        losses: A list to store training losses for each epoch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 device: torch.device, 
                 num_hidden_layers: int = 1,
                 batch_size: int = 32, 
                 l2_reg: float = 0.0,
                 lr: float = 0.001, 
                 weights: Optional[List[float]] = None,
                 random_seed: Optional[int] = None,
                 message: Optional[str] = None):
        """
        Initialize the StaticGNN model.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layers.
            output_dim (int): Dimensionality of the output layer.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
            batch_size (int, optional): Batch size for training. Defaults to 32.
            l2_reg (float, optional): L2 regularization strength. Defaults to 0.0.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            weights (Optional[List[float]], optional): Weights for the output dimensions. Defaults to None.
            random_seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            message (Optional[str], optional): Custom message for the model. Defaults to None.
        """
        super(StaticGNN, self).__init__()

        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.lr = lr
        self.batch_size = batch_size
        self.l2_reg = l2_reg

        if weights is None:
            self.weights = torch.ones(output_dim).to(self.device)
        else:
            assert len(weights) == output_dim, "Length of weights must be equal to output_dim"
            self.weights = torch.tensor(weights).to(self.device)
        
        # layers
        self.convs = torch.nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_hidden_layers):
            self.convs.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        # Model utilities
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.device = device
        self.losses = []
        self.states = []

        self.params = {'model': 'GNN',
                       'input_dim': input_dim,
                       'hidden_dim': hidden_dim,
                       'output_dim': output_dim,
                       'device': str(device),
                       'num_hidden_layers': num_hidden_layers,
                       'lr': lr,
                       'batch_size': batch_size,
                       'l2_reg': l2_reg,
                       'weights': weights,
                       'random_seed': random_seed,
                       'message': message}
        
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
        
        # First SageConv Layer (Outside of loop)
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Using loop for subsequent Linear + SAGEConv layers
        for i in range(1, len(self.convs) - 1, 2):  # step by 2 because we have two layers in each iteration
            # Fully Connected Layer
            x = self.convs[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            # SAGEConv Layer
            x = self.convs[i+1](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # Final fully connected layer
        x = global_mean_pool(x, batch=data.batch)
        x = self.fc(x)
        
        return x



    def train(self, X_train, edge_index, y_train, X_val=None, y_val=None, epochs=100, use_tqdm=True, save_loss=False):
        """
        Train the StaticGNN model.

        Args:
            X_train (np.array): Training node features.
            edge_index (np.array): Edge index for training graph.
            y_train (np.array): Training target values.
            X_val (Optional[np.array], optional): Validation node features. Defaults to None.
            y_val (Optional[np.array], optional): Validation target values. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            use_tqdm (bool, optional): Whether to display a tqdm progress bar. Defaults to True.
            save_loss (bool, optional): Whether to save the training loss. Defaults to False.
        """
        super().train()

        # Data preparation for training
        X_train = torch.tensor(X_train, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.float)
        train_data_list = [Data(x=X_train[i], edge_index=edge_index, y=y_train[i].unsqueeze(
            0)) for i in range(X_train.shape[0])]
        loader = DataLoader(
            train_data_list, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        # Data preparation for validation if provided
        if X_val is not None and y_val is not None:
                X_val = torch.tensor(X_val, dtype=torch.float)
                y_val = torch.tensor(y_val, dtype=torch.float)
                val_data_list = [Data(x=X_val[i], edge_index=edge_index, y=y_val[i].unsqueeze(
                    0)) for i in range(X_val.shape[0])]
                val_loader = DataLoader(val_data_list, batch_size=self.batch_size * 10)

        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=epochs*len(loader),
                        desc="Training Progress")

        for epoch in range(epochs):
            for data in loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self(data)
                loss = self.criterion(out, data.y) * self.weights
                loss.mean().backward()
                self.optimizer.step()

                if use_tqdm:
                    pbar.set_description(f"Epoch {epoch}")
                    pbar.update()
            
            if use_tqdm:
                pbar.set_postfix({'loss': loss.mean().item()})

            if save_loss:
                if X_val is not None and y_val is not None:
                    val_losses = np.zeros((0, y_val.shape[1]))
                    for val_data in val_loader:
                        val_data = val_data.to(self.device)
                        with torch.no_grad():
                            val_out = self(val_data)
                            val_loss = self.criterion(val_out, val_data.y)
                            val_loss = np.mean(val_loss.detach().cpu().numpy(), axis=0)
                            val_losses = np.vstack((val_losses, val_loss))
                    losses = np.mean(val_losses, axis=0)
                    mean_loss = float(np.mean(losses))
                else:
                    losses = np.mean(loss.detach().cpu().numpy(), axis=0)
                    mean_loss = float(np.mean(losses))


                self.losses.append((*losses.tolist(), mean_loss))
                self.states.append(self.state_dict())

        if use_tqdm:
            pbar.close()

        if save_loss:
            self.losses = [self._moving_average(loss).tolist() for loss in np.transpose(self.losses)]

            self.params['losses'] = self.losses
            self.params['best_epoch'] = int(np.argmin(self.losses[-1]))

            # Get the current time and format it as a string
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # Save the parameters and losses with a timestamp in the filename
            with open(f'results/losses_{timestamp}.json', 'w') as f:
                json.dump(self.params, f)
            torch.save(self.states[self.params['best_epoch']], f'models/best_model_{timestamp}.pth')

    def predict(self, X_test, edge_index, y_test):
        """
        Predict with the trained StaticGNN model.

        Args:
            X_test (np.array): Test node features.
            edge_index (np.array): Edge index for test graph.
            y_test (np.array): Test target values.

        Returns:
            np.array: Predicted values for the test data.
        """
        super().train(False)
        X_test = torch.tensor(X_test, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.float)

        test_data_list = [Data(x=X_test[i], edge_index=edge_index, y=y_test[i])
                          for i in range(X_test.shape[0])]
        test_loader = DataLoader(test_data_list, batch_size=1, pin_memory=True)

        predictions = np.zeros((0, 3))
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                pred = self(data)
                predictions = np.vstack((predictions, pred.cpu().numpy()))
        return predictions

    def _moving_average(self, data, window_size=5):
        """ Compute moving average using numpy. """
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')