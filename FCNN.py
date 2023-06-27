import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class TwoInputFCNN:
    def __init__(self, input_dim, hidden_dim, output_dim, device, lr=0.001):
        self.device = device
        self.lr = lr
        self.losses = []

        self.model_R = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)

        self.model_L = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device)

        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def parameters(self):
        return list(self.model_R.parameters()) + list(self.model_L.parameters()) + list(self.fc.parameters())

    def train(self, X_R, X_L, y, epochs=100, batch_size=32, use_tqdm=True, save_loss=False):
        X_R_tensor = torch.tensor(X_R, dtype=torch.float32).to(self.device)
        X_L_tensor = torch.tensor(X_L, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_R_tensor, X_L_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pbar = None
        if use_tqdm:
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
                loss.backward()
                self.optimizer.step()

                if use_tqdm:
                    if step % len(data_loader) == 0:
                        pbar.set_postfix({'loss': loss.item()})

                    pbar.set_description(f"Epoch {epoch}")
                    pbar.update()
                step += 1

            if save_loss:
                self.losses.append((epoch, loss.item()))

        if use_tqdm:
            pbar.close()

        if save_loss:
            np.save('losses.npy', np.array(self.losses))

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
