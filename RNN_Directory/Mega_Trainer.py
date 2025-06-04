import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import torch
import itertools
from RNN_Module import get_data_filtered, collate_fn, CustomDataset, SimpleRNN, model_name, train_model, eval_rnn

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # List of mechanisms for data importing
    mech_list = [("1000", 'M1_cd'), ("0100", 'M1_pd'), ("0010", 'M1_sd'), ("0001", 'M1_n')]

    # List of hyper parameters to test
    activations_list = ['gelu', 'lrelu', 'tanh'] # I do not like vanilla ReLU, i do not think negative inputs should be zeroed
    hidden_size_list = [196, 256] # 128 was too small from previous attempts
    num_layers_list = [4] # Finn suggested 3 and 5, for vibes ?
    rnn_type_list = ["LSTM", "GRU"] # Hopefully this makes all the models as I schnooze, tomorrow we can check GRU

    # training data
    data_list, labels_list = get_data_filtered(mech_list, 0, 8500)
    train_dataset = CustomDataset(data_list, labels_list)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)

    # Evaluate model
    data_list, labels_list = get_data_filtered(mech_list, 8501, 10501)
    test_dataset = CustomDataset(data_list, labels_list)
    test_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)

    for activation, hidden_size, num_layers, rnn_type in itertools.product(activations_list, hidden_size_list, num_layers_list, rnn_type_list):

        # model bizness
        print(f"Training model with: activation={activation}, hidden_size={hidden_size}, layers={num_layers}, type={rnn_type}")
        model = SimpleRNN(input_size=2, hidden_size=hidden_size, num_layers=num_layers, num_classes=4, rnn_type=rnn_type, activation=activation).to(device)

        # optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0075)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train model
        train_model(model, train_loader, criterion, optimizer, device, num_epochs=200)
        model.load_state_dict(torch.load(f"{model_name(model)}"), strict=False)
        model.to(device)
        eval_rnn(model, test_loader, criterion, device)






