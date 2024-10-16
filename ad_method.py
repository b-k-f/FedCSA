import torch
import torch.nn as nn

def flatten_weights(state_dict):
    weights = []
    for key in state_dict:
        weights.append(state_dict[key].cpu().flatten())  # Move tensor to CPU before flattening
    return torch.cat(weights)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomalies_with_autoencoder(model, threshold):
    criterion = nn.MSELoss()
    num_ep = 3
    btch_size = 32
    initial_flattened_weights = flatten_weights(model.state_dict()).numpy().reshape(1, -1)
    ds = torch.tensor(initial_flattened_weights, dtype=torch.float32)
    inpt_dm = ds.shape[1]
    autoencoder = Autoencoder(inpt_dm)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=btch_size, shuffle=True)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

    for epoch in range(num_ep):
        for data in data_loader:
            output = autoencoder(data)
            lss = criterion(output, data)

            optimizer.zero_grad()
            lss.backward()
            optimizer.step()
            
    reconstruction_error = torch.mean((ds - output) ** 2).item()
    is_anomaly = reconstruction_error > threshold
    return is_anomaly, reconstruction_error