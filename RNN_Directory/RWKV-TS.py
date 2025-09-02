import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from RNN_Module import get_data_filtered, CustomDataset


# -----------------------
# Input Module
# -----------------------
class InstanceNorm(nn.Module):
    def forward(self, x):
        # x: [B, L, M]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std, mean, std

class Patcher(nn.Module):
    def __init__(self, in_channels, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)
    def forward(self, x):
        B, L, M = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patches.append(x[:, i:i+self.patch_len, :])  # [B, P, M]
        patches = torch.stack(patches, dim=1)  # [B, N, P, M]
        patches = patches.permute(0, 3, 1, 2)   # [B, M, N, P]
        B, M, N, P = patches.shape
        patches = patches.reshape(B*M, N, P)
        tokens = self.proj(patches)             # [B*M, N, D]
        tokens = tokens.reshape(B, M, N, -1).permute(0, 2, 1, 3) # [B, N, M, D]
        return tokens

# -----------------------
# RWKV-TS Components
# -----------------------
class TimeMixing(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # projections
        self.Wg = nn.Linear(d_model, d_model)
        self.Wr = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        # learnable coefficients
        self.mg = nn.Parameter(torch.rand(d_model))
        self.mr = nn.Parameter(torch.rand(d_model))
        self.mk = nn.Parameter(torch.rand(d_model))
        self.mv = nn.Parameter(torch.rand(d_model))
        self.w = nn.Parameter(torch.randn(self.n_heads, self.head_dim))
        self.u = nn.Parameter(torch.randn(self.n_heads, self.head_dim))

    def forward(self, x, mask=None):
        # x: [B, N, D]
        B, N, D = x.shape
        x_shift = torch.roll(x, shifts=1, dims=1)
        g = self.Wg(self.mg * x + (1 - self.mg) * x_shift)
        r = self.Wr(self.mr * x + (1 - self.mr) * x_shift)
        k = self.Wk(self.mk * x + (1 - self.mk) * x_shift)
        v = self.Wv(self.mv * x + (1 - self.mv) * x_shift)

        # multi-head split
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, N, Hd]
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        r = r.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        g = g.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        decay = torch.exp(-torch.exp(self.w))  # [H, Hd]
        u = self.u
        wkv = []
        for h in range(self.n_heads):
            acc = torch.zeros(B, self.head_dim, device=x.device)
            out_h = []
            for t in range(N):
                acc = decay[h] * acc + k[:, h, t, :] * v[:, h, t, :]
                out_h.append(acc + u[h] * (k[:, h, t, :] * v[:, h, t, :]))
            out_h = torch.stack(out_h, dim=1)  # [B, N, Hd]
            wkv.append(out_h)
        wkv = torch.stack(wkv, dim=1)  # [B, H, N, Hd]

        out = torch.sigmoid(g) * F.layer_norm(r * wkv, (self.head_dim,))
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.Wo(out)

class ChannelMixing(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wg = nn.Linear(d_model, d_model)
        self.Wr = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.mk = nn.Parameter(torch.rand(d_model))
        self.mr = nn.Parameter(torch.rand(d_model))
    def forward(self, x, mask=None):
        x_shift = torch.roll(x, shifts=1, dims=1)
        k = self.Wg(self.mk * x + (1-self.mk) * x_shift)
        r = self.Wr(self.mr * x + (1-self.mr) * x_shift)
        v = self.Wv(F.relu(k)**2)
        return torch.sigmoid(r) * v

class RWKVBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.time_mix = TimeMixing(d_model, n_heads)
        self.chan_mix = ChannelMixing(d_model)

    def forward(self, x, mask=None):  # <--- add mask here
        x = x + self.time_mix(x, mask)
        x = x + self.chan_mix(x, mask)
        return x

# -----------------------
# Full Model
# -----------------------
class RWKV_TS(nn.Module):
    def __init__(self, in_channels, d_model=128, num_layers=2, patch_len=16, stride=8, out_dim=1, n_heads=2, bidirectional=False):
        super().__init__()
        self.norm = InstanceNorm()
        self.patcher = Patcher(in_channels, patch_len, stride, d_model)
        self.layers_fwd = nn.ModuleList([RWKVBlock(d_model, n_heads) for _ in range(num_layers)])
        self.bidirectional = bidirectional
        if bidirectional:
            self.layers_bwd = nn.ModuleList([RWKVBlock(d_model, n_heads) for _ in range(num_layers)])
            self.head = nn.Linear(d_model*2, out_dim)
        else:
            self.head = nn.Linear(d_model, out_dim)

    def forward_one_direction(self, tokens, layers, mask=None):
        for blk in layers:
            tokens = blk(tokens, mask)
        return tokens

    def forward(self, x, lengths=None):
        x, mean, std = self.norm(x)
        tokens = self.patcher(x)  # [B,N,M,D]
        B, N, M, D = tokens.shape
        tokens = tokens.reshape(B*M, N, D)

        # mask logic
        if lengths is not None:
            patch_scale = tokens.shape[1] / x.shape[1]
            patch_lengths = torch.clamp((lengths.float() * patch_scale).long(), max=tokens.shape[1])
            mask = torch.arange(tokens.shape[1], device=x.device)[None, :].expand(B*M, -1)
            valid_mask = mask < patch_lengths.repeat_interleave(M, dim=0)[:, None]
        else:
            valid_mask = None

        out_fwd = self.forward_one_direction(tokens, self.layers_fwd, valid_mask)
        if self.bidirectional:
            tokens_rev = torch.flip(tokens, dims=[1])
            out_bwd = self.forward_one_direction(tokens_rev, self.layers_bwd, valid_mask)
            out_bwd = torch.flip(out_bwd, dims=[1])
            out_combined = torch.cat([out_fwd.mean(1), out_bwd.mean(1)], dim=-1)
        else:
            out_combined = out_fwd.mean(1)

        out_combined = out_combined.view(B, M, -1).mean(dim=1)
        out = self.head(out_combined)
        
        return out



def collate_fn_rwkv(batch):
    # batch: list of (seq[L,M], label)
    sequences, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)  # [B, L_max, M]
    labels = torch.stack(labels)                        # [B, num_classes]
    return padded, labels, lengths

def train_model_rwkv(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_elements = 0

        for sequences, labels, _ in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(sequences)  # RWKV_TS does not need lengths
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ---- Accuracy computation ----
            probs = torch.sigmoid(outputs)               # convert logits to probabilities
            preds = (probs >= 0.5).float()               # threshold
            correct = (preds == labels).sum().item()     # element-wise match
            total_correct += correct
            total_elements += labels.numel()             # total number of label elements

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_elements if total_elements > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")


def eval_rwkv(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            lengths = lengths.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# -----------------------
# Example training loop
# -----------------------
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")

    model = RWKV_TS(
        in_channels=2,
        d_model=128,
        num_layers=2,
        patch_len=16,
        stride=4,
        out_dim=4,
        n_heads=4,
        bidirectional=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # List of mechanisms for data importing
    # mech_list = [("10000", 'M1_cd'), ("01000", 'M1_pd'), ("00100", 'M1_sd'), ("00010", 'M1_n'), ('00001', 'M1_si')]
    mech_list = [("1000", 'M1_cd'), ("0100", 'M1_pd'), ("0010", 'M1_sd'), ("0001", 'M1_n')]
    # Training data
    data_list, labels_list = get_data_filtered(mech_list, 0, 8500)
    print(f"Total training sequences loaded: {len(data_list)}")
    train_dataset = CustomDataset(data_list, labels_list)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn_rwkv)

    # Train
    train_model_rwkv(model, train_loader, criterion, optimizer, device, num_epochs=350)
    torch.save(model.state_dict(), "rwkvts_model2.pth")

    # Test data
    test_data, test_labels = get_data_filtered(mech_list, 8501, 10501)
    print(f"Total test sequences loaded: {len(test_data)}")
    test_dataset = CustomDataset(test_data, test_labels)
    test_loader  = DataLoader(test_dataset, batch_size=2048, shuffle=False, collate_fn=collate_fn_rwkv)

    # Evaluate
    eval_rwkv(model, test_loader, criterion, device)
