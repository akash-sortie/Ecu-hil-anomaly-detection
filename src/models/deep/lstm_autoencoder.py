"""
LSTM Autoencoder for ECU HIL Signal Anomaly Detection
Architecture: Encoder LSTM → Bottleneck → Decoder LSTM → Linear projection
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # (batch, hidden_size)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, seq_len: int, n_features: int, num_layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, z):
        # z: (batch, hidden_size) → repeat across seq_len
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_repeated)
        return self.output_layer(out)  # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=12, hidden_size=64, num_layers=2, seq_len=50, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, seq_len, n_features, num_layers, dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        """Per-sample mean squared reconstruction error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2))


def train(config: dict, train_loader, val_loader):
    device = torch.device(config.get("device", "cpu"))
    model = LSTMAutoencoder(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        seq_len=config["seq_len"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    mlflow.set_experiment("ECU_HIL_LSTM_AE")
    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch[0].to(device)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.6f}")

        # Compute threshold from validation reconstruction errors
        model.eval()
        val_errors = []
        for batch in val_loader:
            errors = model.reconstruction_error(batch[0].to(device))
            val_errors.extend(errors.cpu().numpy())

        threshold = np.percentile(val_errors, config["threshold_percentile"])
        mlflow.log_metric("anomaly_threshold", threshold)
        mlflow.pytorch.log_model(model, "lstm_autoencoder")
        print(f"Anomaly threshold (p{config['threshold_percentile']}): {threshold:.6f}")

    return model, threshold
