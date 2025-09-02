import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
from RNN_Module import get_data_filtered, collate_fn, CustomDataset, collate_pdn
from RNN_Module import model_name, train_model_rnn, train_model_tcn, eval_rnn, eval_tcn, GRUxLSTMx, TCN

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    # List of mechanisms for data importing
    mech_list = [("10000", 'M1_cd'), ("01000", 'M1_pd'), ("00100", 'M1_sd'), ("00010", 'M1_n'), ('00001', 'M1_si')]
    #mech_list = [("1000", 'M1_cd'), ("0100", 'M1_pd'), ("0010", 'M1_sd'), ("0001", 'M1_n')]

    # training data
    data_list, labels_list = get_data_filtered(mech_list, 0, 9000)
    print(f"Total sequences loaded: {len(data_list)}")
    train_dataset = CustomDataset(data_list, labels_list)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    
    # # TCN is after this part
    # model = TCN(input_size=2, num_channels=[256]*4, kernel_size = 3, dropout=0.2, num_classes=5).to(device)


    # GRUxLSTMx is after this part
    model = GRUxLSTMx(input_size=2, hidden_size=64, num_gru=3, num_lstm=1, num_classes=5, activation="relu").to(device)

    # optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training
    train_model_rnn(model, train_loader, criterion, optimizer, device, num_epochs=300)
    # Eval
    data_list, labels_list = get_data_filtered(mech_list, 9001, 10501)
    print(f"Total sequences loaded: {len(data_list)}")
    test_dataset = CustomDataset(data_list, labels_list)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    model.load_state_dict(torch.load("MODEL_20250808_033432.pth"), strict=False)
    model.to(device)
    eval_rnn(model, test_loader, criterion, device)







