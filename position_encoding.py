"""
Task: Design a learnable positional encoding method using Pytorch

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DummyData(Dataset):
    """
    Create dummmy dataset
    """
    def __init__(self, n_seqs, n_samples, n_features: int = 1):
        """
        :param n_seqs: the length of the sequence length
        :param n_samples: the number of the samples in the dataset
        :param n_features: the number of the features of the dataset
        """
        self.n_seqs = n_seqs
        self.n_samples = n_samples
        self.n_features = n_features
        self.seqs = torch.randn(n_samples, n_seqs, n_features)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.seqs[idx], self.seqs[idx]


# Define the learnable positional encoding module
class ModelPe(nn.Module):
    def __init__(self, n_seq: int = 100, n_dims: int = 1, n_out: int = 20):
        """
        :param n_seq: the length of the sequence
        :param n_dims: the dimension of each training instance
        :param n_out: the length of the output of the first layer
        """
        super(ModelPe, self).__init__()
        # Define the learnable positional encoding embedding layer
        self.embedding = nn.Embedding(n_seq+1, n_dims, padding_idx=0)
        # Add two more linear layers for demonstration purpose. Any number of the layers could be added here
        self.linear1 = nn.Linear(n_seq+1, n_out)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(n_out, n_seq)

    def forward(self, x):
        batch_size, n_seq, _ = x.size()
        # Generate the positional vector. Please note that I added one extra learnable embedding node here at 0 location
        positions = torch.arange(0, n_seq+1)
        positions = positions.unsqueeze(0)
        positions = positions.repeat(batch_size, 1)
        # Generate the positional encoding results from the learnable embedding layer
        position_encoding = self.embedding(positions)
        # Padding x with 0 at the beginning of the first dimension
        x = nn.functional.pad(x, pad=(0, 0, 1, 0), mode='constant', value=0)
        x = x + position_encoding
        x = x.squeeze()
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def main():
    # Define hyperparameters
    n_seq = 10  # The length of the sequence length
    n_samples = 200  # The number of the samples
    n_epochs = 5  # The epoch number
    lr = 0.003  # The learning rate
    batch_size = 2  # The batch size of the training set

    # Create dummy dataset
    data_set = DummyData(n_seq, n_samples)

    # Create training set
    training_data = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = ModelPe(n_seq)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Model training loop
    for epoch in range(n_epochs):
        sum_loss = 0
        for x_train, y_train in training_data:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss_tmp = loss(y_pred, y_train.squeeze())
            loss_tmp.backward()
            optimizer.step()
            sum_loss += loss_tmp.item()
        print(f'The loss is {sum_loss} at Epoch {epoch + 1}')


if __name__ == '__main__':
    main()