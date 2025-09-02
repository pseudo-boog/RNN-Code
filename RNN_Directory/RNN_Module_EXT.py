from tqdm import tqdm
import time, os
import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

class MixedRNN(nn.Module):
    def __init__(self, input_size_x1, input_size_x2, hidden_size, num_gru, num_lstm, num_classes, activation="relu"):
        super().__init__()
        self.fc_x1 = nn.Linear(input_size_x1, hidden_size)

        activation = activation.lower()
        activations = {
            "relu": nn.ReLU(), "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(), "gelu": nn.GELU(),
            "lrelu": nn.LeakyReLU()
        }
        self.activation = activations[activation]

        self.rnns = nn.ModuleList()
        for i in range(num_gru):
            in_dim = input_size_x2 if i == 0 else hidden_size
            self.rnns.append(nn.GRU(in_dim, hidden_size, batch_first=True))
        for i in range(num_lstm):
            in_dim = input_size_x2 if (num_gru + i == 0) else hidden_size
            self.rnns.append(nn.LSTM(in_dim, hidden_size, batch_first=True))

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(hidden_size + hidden_size, num_classes)  # concat x1 and RNN out

    def forward(self, x1, x2, lengths):
        packed = pack_padded_sequence(x2, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for rnn in self.rnns:
            packed, _ = rnn(packed)
            out, _ = pad_packed_sequence(packed, batch_first=True)
            out = self.activation(out)
            packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, _ = pad_packed_sequence(packed, batch_first=True)
        out = self.batch_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_out = out.gather(1, idx).squeeze(1)

        x1_feat = self.activation(self.fc_x1(x1))
        combined = torch.cat([x1_feat, last_out], dim=1)
        return self.fc_out(combined)

class VariableTPDataset(Dataset):
    def __init__(self, x1_path, x2_path, y_path, min_T=3, rng_seed=42):
        with open(x1_path, "rb") as f:
            self.x1 = pickle.load(f).astype(np.float32)

        with open(x2_path, "rb") as f:
            x2_raw = pickle.load(f)

        with open(y_path, "rb") as f:
            self.y = pickle.load(f).astype(np.float32)

        self.min_T = int(min_T)
        self.rng = np.random.default_rng(rng_seed)

        # Normalize x2 into a dict {T: (N, T, F)}
        self.x2_by_T = self._normalize_x2_variants(x2_raw)
        self.T_values = sorted(self.x2_by_T.keys())          # e.g., [21] or [3,7,21]
        self.Tmax = int(max(self.T_values))                   # largest T available
        # Allow sampling *any* T in [min_T .. Tmax]
        self.all_T_choices = list(range(self.min_T, self.Tmax + 1))

    def _normalize_x2_variants(self, x2_raw):
        # If ndarray, wrap in dict with its T
        if isinstance(x2_raw, np.ndarray):
            assert x2_raw.ndim == 3, f"x2 must be (N,T,F); got {x2_raw.shape}"
            T = int(x2_raw.shape[1])
            return {T: x2_raw.astype(np.float32)}

        # Dict format (possibly nested dict)
        x2_by_T = {}
        for k, v in x2_raw.items():
            # Your test files used outer key as (T-1)
            try:
                T = int(k) + 1
            except Exception:
                try:
                    T = int(k)
                except Exception:
                    continue
            arr = None
            if isinstance(v, dict):
                # choose a stable inner entry
                ik = sorted(v.keys())[0]
                arr = v[ik]
            else:
                arr = v
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                x2_by_T[int(T)] = arr.astype(np.float32)
        if not x2_by_T:
            raise ValueError("Could not normalize x2 variants into {T: (N,T,F)}.")
        return x2_by_T

    def __len__(self):
        return self.x1.shape[0]

    def _get_x2_T(self, T):
        if T in self.x2_by_T:
            return self.x2_by_T[T]
        larger = sorted(t for t in self.T_values if t >= T)
        if not larger:
            raise ValueError(f"No available sequence >= T={T}. Have {self.T_values}.")
        Tbig = larger[0]
        return self.x2_by_T[Tbig][:, :T, :]

    def __getitem__(self, idx):
        # Pick any T in [min_T..Tmax], then get that length (or slice from a longer one)
        T_choice = int(self.rng.choice(self.all_T_choices))
        x2_full = self._get_x2_T(T_choice)     # (N, T_choice, F)
        x2 = torch.tensor(x2_full[idx], dtype=torch.float32)
        length = x2.shape[0]

        x1 = torch.tensor(self.x1[idx], dtype=torch.float32)
        y  = torch.tensor(self.y[idx],  dtype=torch.float32)

        return x1, x2, length, y

def collate_fn(batch):
    x1_list, x2_list, lengths, y_list = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    x1_batch = torch.stack(x1_list, dim=0)
    x2_padded = pad_sequence(x2_list, batch_first=True)  # (B, Tmax_in_batch, F)
    y_batch = torch.stack(y_list, dim=0)
    return x1_batch, x2_padded, lengths, y_batch

def train_loop(model, loader, criterion, optimizer, device, epochs=150, val_loader=None):
    ckpt_dir = "./checkpoints"
    ckpt_prefix = "RNN_EPIC"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_acc = -1.0

    print(">>> starting training", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x1, x2, lengths, y in pbar:
            x1, x2, lengths, y = x1.to(device), x2.to(device), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y.argmax(dim=1)).sum().item()
            total += y.size(0)

            pbar.set_postfix({
                "loss": f"{loss_sum / max(1, total):.4f}",
                "acc":  f"{correct / max(1, total):.4f}"
            })

        train_loss = loss_sum / max(1, total)
        train_acc  = correct / max(1, total)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        # Save checkpoints
        last_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_last.pth")
        torch.save(model.state_dict(), last_path)
        saved_note = ""
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best.pth")
            torch.save(model.state_dict(), best_path)
            saved_note = " [saved best]"

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}{saved_note}", flush=True)

    dur_min = (time.time() - start_time) / 60
    print(f"Training finished in {dur_min:.1f} min. Best val acc: {best_val_acc:.4f}", flush=True)
    print(f"Checkpoints: {ckpt_dir}/{ckpt_prefix}_last.pth"
          f"{' and ' + ckpt_dir + '/' + ckpt_prefix + '_best.pth' if val_loader is not None else ''}",
          flush=True)

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x1, x2, lengths, y in loader:
        x1, x2, lengths, y = x1.to(device), x2.to(device), lengths.to(device), y.to(device)
        logits = model(x1, x2, lengths)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y.argmax(dim=1)).sum().item()
        total += y.size(0)
    avg_loss = (loss_sum / total) if criterion is not None and total > 0 else float('nan')
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def run():
    # These are the file paths to be used in the test on my local machine. 
    # change as needed. These are the data sets from Jordi Bures
    X1_PATH = "/Users/dylanpyle/VsCode/RNN_Code/DATA/TESTING/x1_train_M1_M20_train_val_test_set.pkl"
    X2_PATH = "/Users/dylanpyle/VsCode/RNN_Code/DATA/TESTING/x2_train_M1_M20_train_val_test_set.pkl"
    Y_PATH  = "/Users/dylanpyle/VsCode/RNN_Code/DATA/TESTING/y_train_M1_M20_train_val_test_set.pkl"

    EPOCHS      = 50 # number of epochs
    BATCH_SIZE  = 2048 # batch size
    LR          = 1e-3 # learning rate
    HIDDEN_SIZE = 128 # hidden size of RNN layers
    NUM_GRU     = 1 # number of GRU layers
    NUM_LSTM    = 2 # number of LSTM layers
    MIN_T       = 3 # minimum sequence length to sample

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = VariableTPDataset(X1_PATH, X2_PATH, Y_PATH, min_T=MIN_T)

    # one-hot encode labels i
    import numpy as np

    if isinstance(dataset.y, np.ndarray):
        if dataset.y.ndim == 1 or (dataset.y.ndim == 2 and dataset.y.shape[1] == 1):
            y_idx = dataset.y.reshape(-1).astype(np.int64)
            num_classes = int(y_idx.max()) + 1
            y_oh = np.zeros((y_idx.shape[0], num_classes), dtype=np.float32)
            y_oh[np.arange(y_idx.shape[0]), y_idx] = 1.0
            dataset.y = y_oh
        else:
            num_classes = dataset.y.shape[1]
            if dataset.y.dtype != np.float32:
                dataset.y = dataset.y.astype(np.float32)
    else:
        raise TypeError(f"Unsupported label container: {type(dataset.y).__name__}")

    # DataLoader with workers
    import os
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=True,
    )

    model = MixedRNN(
        input_size_x1=dataset.x1.shape[1],
        input_size_x2=dataset.x2_by_T[next(iter(dataset.x2_by_T))].shape[2],
        hidden_size=HIDDEN_SIZE,
        num_gru=NUM_GRU,
        num_lstm=NUM_LSTM,
        num_classes=num_classes,
        activation="relu",
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(">>> starting training", flush=True)
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_loop(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}", flush=True)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    print(">>> starting RNN_EPIC.py", flush=True)
    run()
