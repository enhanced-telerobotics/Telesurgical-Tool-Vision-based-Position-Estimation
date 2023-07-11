import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime


class TwoInputFCNN:
    def __init__(self, input_dim, hidden_dim, output_dim, device, num_hidden_layers=2, lr=0.001, batch_size=32):
        self.device = device
        self.lr = lr
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.losses = []
        self.params = {'input_dim': input_dim,
                       'hidden_dim': hidden_dim,
                       'output_dim': output_dim,
                       'device': str(device),
                       'num_hidden_layers': num_hidden_layers,
                       'lr': lr,
                       'batch_size': batch_size}

        self.model_R = self._create_hidden_layers(
            input_dim, hidden_dim, num_hidden_layers).to(self.device)
        self.model_L = self._create_hidden_layers(
            input_dim, hidden_dim, num_hidden_layers).to(self.device)

        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(self.device)

        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _create_hidden_layers(self, input_dim, hidden_dim, num_layers):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        return nn.Sequential(*layers)

    def parameters(self):
        return list(self.model_R.parameters()) + list(self.model_L.parameters()) + list(self.fc.parameters())

    def train(self, X_R, X_L, y, X_val_R=None, X_val_L=None, y_val=None, epochs=100, use_tqdm=True, save_loss=False):
        X_R_tensor = torch.tensor(X_R, dtype=torch.float32).to(self.device)
        X_L_tensor = torch.tensor(X_L, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        if X_val_R is not None and X_val_L is not None and y_val is not None:
            X_val_R_tensor = torch.tensor(
                X_val_R, dtype=torch.float32).to(self.device)
            X_val_L_tensor = torch.tensor(
                X_val_L, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(
                y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_R_tensor, X_L_tensor, y_tensor)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=epochs*len(data_loader),
                        desc="Training Progress")
            step = 0

        for epoch in range(epochs):
            for X_R_batch, X_L_batch, y_batch in data_loader:
                self.optimizer.zero_grad()
                out_R = self.model_R(X_R_batch)
                out_L = self.model_L(X_L_batch)
                out = torch.cat((out_R, out_L), dim=1)
                y_pred = self.fc(out)
                loss = self.criterion(y_pred, y_batch)
                loss.mean().backward()
                self.optimizer.step()

                if use_tqdm:
                    pbar.set_description(f"Epoch {epoch}")
                    pbar.update()
                    step += 1

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
                            y_val_pred, y_val_tensor).detach().cpu().numpy()

                        # Save the per-dimension validation loss and the mean validation loss
                        self.losses.append(
                            (*np.mean(val_loss, axis=0).tolist(), float(np.mean(val_loss))))
                else:
                    # Save the per-dimension loss and the mean loss
                    losses = loss.detach().cpu().numpy().tolist()
                    self.losses.append(
                        (*np.mean(losses, axis=0).tolist(), float(np.mean(losses))))


        if use_tqdm:
            pbar.close()

        if save_loss:
            self.params['losses'] = self.losses

            # Get the current time and format it as a string
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            # Save the parameters and losses with a timestamp in the filename
            with open(f'results/params_and_losses_{timestamp}.json', 'w') as f:
                json.dump(self.params, f)

    def predict(self, X_R, X_L):
        X_R_tensor = torch.tensor(X_R, dtype=torch.float32).to(self.device)
        X_L_tensor = torch.tensor(X_L, dtype=torch.float32).to(self.device)
        out_R = self.model_R(X_R_tensor)
        out_L = self.model_L(X_L_tensor)
        out = torch.cat((out_R, out_L), dim=1)
        return self.fc(out).cpu().detach().numpy()


if __name__ == '__main__':
    pass
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = TwoInputFCNN(input_dim=32, hidden_dim=64, output_dim=3, device=device)  # changed output_dim to 3
    # net.train(X_train_R, X_train_L, y_train, epochs=100)
    # y_pred = net.predict(X_test_R, X_test_L)
