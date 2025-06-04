import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import torch

def get_data_filtered(mech_list, start_iteration, end_iteration):

    data_list = []
    labels_list = []
    num_features = 2  # A, P
    min_time_points = 15

    # Just for logging:
    total_found = 0
    total_skipped = 0

    for i in range(start_iteration, end_iteration + 1):
        for j in range(2):  # Adjust as necessary
            for label, prefix in mech_list:

                file_path = f'/Users/dylanpyle/PycharmProjects/code/DATA/Training/{prefix}/rct_{i}/exp_{j}.csv'

                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}. Skipping.")
                    continue

                try:
                    df = pd.read_csv(file_path, skiprows=1, header=None, usecols=[0, 1])
                    df.columns = ['A', 'P']
                    df = df.dropna().select_dtypes(include=[np.number])
                    data_array = df.values

                    # Basic shape checks
                    if data_array.size == 0 or data_array.shape[1] != num_features:
                        print(f"Invalid data shape in {file_path}: {data_array.shape}")
                        continue

                    # NEW: Filter out sets with fewer than 15 time points
                    if data_array.shape[0] < min_time_points:
                        print(f"Skipping {file_path} because it has fewer than {min_time_points} time points.")
                        total_skipped += 1
                        continue

                    # Otherwise, valid data
                    data_tensor = torch.tensor(data_array, dtype=torch.float32)
                    data_list.append(data_tensor)
                    labels_list.append(label)
                    total_found += 1

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}. Skipping.")
                    continue

    if len(data_list) == 0 or len(labels_list) == 0:
        raise ValueError("No valid data or labels found. Check your file paths and data processing.")

    print(f"\nFinished loading data. Found {total_found} valid datasets. Skipped {total_skipped} datasets.")
    return data_list, labels_list

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Pad the sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # Convert each label string into ints
    numeric_labels = []
    for label_str in labels:
        label_array = [int(ch) for ch in label_str]
        numeric_labels.append(label_array)

    # Make a float tensor for BCEWithLogitsLoss
    numeric_labels = torch.tensor(numeric_labels, dtype=torch.float)

    return padded_sequences, numeric_labels, lengths

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int = 4, rnn_type: str = "LSTM", activation: str = "relu"):
        super(SimpleRNN, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_classes = num_classes

        # choose RNN cell class
        self.rnn_type = rnn_type.upper()
        rnn_cls = {"LSTM": nn.LSTM, "GRU":  nn.GRU, "RNN":  nn.RNN}[self.rnn_type]

        # build num_layers of RNNs
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.rnns.append(
                rnn_cls(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    batch_first=True
                )
            )

        # Activation Variation
        activation = activation.lower()
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "gelu": nn.GELU(), "lrelu": nn.LeakyReLU()}
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        # batch‐norm & dropout
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout    = nn.Dropout(0.5)

        # final classifier
        self.fc = nn.Linear(hidden_size, num_classes)

        # now initialize all weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each RNN layer’s weights/biases
        for rnn in self.rnns:
            for name, param in rnn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

        # Initialize the final fully‐connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, lengths):
        # ensure CPU ints for pack_padded_sequence
        lengths = lengths.cpu().long()

        # pack once up front
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # pass through each RNN + activation
        for rnn in self.rnns:
            packed, _ = rnn(packed)
            output, _ = pad_packed_sequence(packed, batch_first=True)
            output    = self.activation(output)
            packed    = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)

        # unpack final layer
        output, _ = pad_packed_sequence(packed, batch_first=True)

        # batch‐norm & dropout on feature dim
        output = self.batch_norm(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(output)

        # gather last valid timestep
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2)).to(output.device)
        last_output = output.gather(1, idx).squeeze(1)

        # final linear classifier
        logits = self.fc(last_output)
        return logits

def model_name(model):
    activation = model.activation.__class__.__name__
    return f"{model.rnn_type}_act-{activation}_H{model.hidden_size}_L{model.num_layers}.pth"

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    epoch_list = []
    epoch_loss_list = []
    epoch_accuracy_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (sequences, labels, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            lengths = lengths.cpu().long()  # Move lengths to CPU

            # Forward pass
            outputs = model(sequences, lengths)

            # Compute loss
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            epoch_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = (probabilities >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.numel()

        epoch_loss /= len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_list.append(epoch+1)
        epoch_loss_list.append(epoch_loss)
        epoch_accuracy_list.append(epoch_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], , Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print(epoch_list)
    # Save the model post training, I usually don't do this because I terminate early unless it looks good
    torch.save(model.state_dict(), f"{model_name(model)}") # 5 layers, 128 hidden size, 2 input size
    plt.scatter(epoch_list, epoch_loss_list)
    plt.title("Loss vs Epochs")
    plt.show()
    plt.scatter(epoch_list, epoch_accuracy_list)
    plt.title(f"Accuracy vs Epochs, {model_name(model)}")
    plt.show()
    print(f"Model saved to '{model_name(model)}'")
    return epoch_list, epoch_loss_list, epoch_accuracy_list

def eval_rnn(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            # Move to device
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            lengths = lengths.to(device)

            # Forward pass
            outputs = model(sequences, lengths)  # Shape [batch_size, 1]

            # Compute loss
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # Convert logits to probabilities and then binary predictions
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities >= 0.5).float()

            # Ensure proper shape
            labels = labels.view(-1)  # Flatten labels if needed
            predictions = predictions.view(-1)  # Flatten predictions

            # Counts
            correct += (predictions == labels).sum().item()
            total += labels.numel()  # Ensure total counts actual elements

    # Avg loss and Accuracy
    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0  # Avoid division by zero

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


