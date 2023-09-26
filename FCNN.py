import os
import numpy as np
import json
from datetime import datetime
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List


class TwoInputFCNN:
    """
    A Fully Connected Neural Network (FCNN) model with two inputs.

    This class encapsulates the creation, training, and inference stages of a FCNN model with two inputs.
    The model uses Mean Squared Error (MSE) as the loss function and Adam as the optimizer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 device: torch.device,
                 num_hidden_layers: int = 2,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 l2_reg: float = 0.0,
                 weights: Optional[List[float]] = None,
                 random_seed: Optional[int] = None,
                 message: Optional[str] = None):
        """
        Initialize the TwoInputFCNN model.

        :param input_dim: The dimensionality of the input data.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param output_dim: The dimensionality of the output data.
        :param device: The device to run the model on (cpu or gpu).
        :param num_hidden_layers: The number of hidden layers in the model.
        :param lr: The learning rate for the optimizer.
        :param batch_size: The batch size for training.
        :param l2_reg: The L2 regularization strength.
        :param weights: The weights for the output dimensions.
        :param random_seed: The random seed for reproducibility.
        :param message: Custom message for the model. Defaults to None.
        """        
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

        self.losses = []
        self.states = []
        self.params = {'model': 'FCNN',
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

        # Set the random seed if it's provided
        if random_seed is not None:
            os.environ['PYTHONHASHSEED']=str(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.model_R = self._create_hidden_layers(
            input_dim, hidden_dim, num_hidden_layers).to(self.device)
        self.model_L = self._create_hidden_layers(
            input_dim, hidden_dim, num_hidden_layers).to(self.device)

        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(self.device)

        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=l2_reg)

    def _create_hidden_layers(self, input_dim, hidden_dim, num_layers):
        """
        Create the hidden layers for the model.

        :param input_dim: The dimensionality of the input data.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of hidden layers.
        :return: A Sequential model containing the hidden layers.
        """
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        return nn.Sequential(*layers)

    def parameters(self):
        """
        Get the parameters of the model.

        :return: The parameters of the model.
        """
        return list(self.model_R.parameters()) + list(self.model_L.parameters()) + list(self.fc.parameters())

    def train(self, X_R, X_L, y, X_val_R=None, X_val_L=None, y_val=None, epochs=100, use_tqdm=True, save_loss=False):
        """
        Train the model.

        :param X_R: The right input data for training.
        :param X_L: The left input data for training.
        :param y: The output data for training.
        :param X_val_R: The right input data for validation.
        :param X_val_L: The left input data for validation.
        :param y_val: The output data for validation.
        :param epochs: The number of epochs to train for.
        :param use_tqdm: Whether to use tqdm for progress bar.
        :param save_loss: Whether to save the loss.
        """
        X_R_tensor = torch.tensor(X_R, dtype=torch.float32)
        X_L_tensor = torch.tensor(X_L, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if X_val_R is not None and X_val_L is not None and y_val is not None:
            X_val_R_tensor = torch.tensor(
                X_val_R, dtype=torch.float32).to(self.device)
            X_val_L_tensor = torch.tensor(
                X_val_L, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(
                y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_R_tensor, X_L_tensor, y_tensor)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=epochs*len(data_loader),
                        desc="Training Progress")

        for epoch in range(epochs):
            train_loss = np.zeros(y.shape[1])
            for X_R_batch, X_L_batch, y_batch in data_loader:
                X_R_batch = X_R_batch.to(self.device)
                X_L_batch = X_L_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                out_R = self.model_R(X_R_batch)
                out_L = self.model_L(X_L_batch)
                out = torch.cat((out_R, out_L), dim=1)
                y_pred = self.fc(out)
                loss = self.criterion(y_pred, y_batch) * self.weights
                loss.mean().backward()
                self.optimizer.step()
                train_loss += np.mean(loss.detach().cpu().numpy(), axis=0)

                if use_tqdm:
                    pbar.set_description(f"Epoch {epoch}")
                    pbar.update()

            if use_tqdm:
                pbar.set_postfix({'loss': loss.mean().item()})

            if save_loss:
                if X_val_R is not None and X_val_L is not None and y_val is not None:
                    with torch.no_grad():
                        out_val_R = self.model_R(X_val_R_tensor)
                        out_val_L = self.model_L(X_val_L_tensor)
                        out_val = torch.cat((out_val_R, out_val_L), dim=1)
                        y_val_pred = self.fc(out_val)
                        val_loss = self.criterion(
                            y_val_pred, y_val_tensor)
                        val_loss = val_loss.detach().cpu().numpy()
                        
                        losses = np.mean(val_loss, axis=0)
                        mean_loss = float(np.mean(losses))
                else:
                    # Save the per-dimension loss and the mean loss
                    losses = train_loss/len(data_loader)
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

    def predict(self, X_R, X_L):
        """
        Predict the output given the input data.

        :param X_R: The right input data.
        :param X_L: The left input data.
        :return: The predicted output data.
        """
        X_R_tensor = torch.tensor(X_R, dtype=torch.float32).to(self.device)
        X_L_tensor = torch.tensor(X_L, dtype=torch.float32).to(self.device)
        out_R = self.model_R(X_R_tensor)
        out_L = self.model_L(X_L_tensor)
        out = torch.cat((out_R, out_L), dim=1)
        return self.fc(out).cpu().detach().numpy()
    
    def state_dict(self):
        """
        Get the state dict of the model.

        :return: The state dict of the model.
        """
        return {
            'model_R': self.model_R.state_dict(),
            'model_L': self.model_L.state_dict(),
            'fc': self.fc.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """
        Load the state dict into the model.

        :param state_dict: The state dict to load.
        """
        self.model_R.load_state_dict(state_dict['model_R'])
        self.model_L.load_state_dict(state_dict['model_L'])
        self.fc.load_state_dict(state_dict['fc'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _moving_average(self, data, window_size=5):
        """ Compute moving average using numpy. """
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')


if __name__ == '__main__':
    # Sample usage:
    pass
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = TwoInputFCNN(input_dim=32, hidden_dim=64, output_dim=3, device=device)  # changed output_dim to 3
    # net.train(X_train_R, X_train_L, y_train, epochs=100)
    # y_pred = net.predict(X_test_R, X_test_L)
