# train_stockfish_chunks.py
# Entraînement sur dataset HalfKP encodé en version sparse
# Chaque fichier contient: {"idx": [tensor(int32)...], "y": tensor(float16)}
# Reconstitution dense à la volée pendant l'entraînement

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nnue_core import TinyNNUE


# --- Dataset Sparse à la volée ---
class SparseChunkDataset(Dataset):
    def __init__(self, idx_list, y_tensor, input_dim=40960):
        self.idx_list = idx_list
        self.y = y_tensor.to(torch.float32)
        self.input_dim = input_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        idx = self.idx_list[i]
        x = torch.zeros(self.input_dim, dtype=torch.float32)
        x[idx] = 1.0
        return x, self.y[i].unsqueeze(0)


# --- Entraînement ---
def train():
    ENCODED_DIR = "encoded_sparse_chunks"
    CHECKPOINT_DIR = "checkpoints"
    EPOCHS = 5
    BATCH_SIZE = 256
    LR = 1e-3
    INPUT_DIM = 40960

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Entraînement sur {device}")

    chunks = sorted(glob.glob(os.path.join(ENCODED_DIR, "encoded_sparse_*.pt")))
    if not chunks:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {ENCODED_DIR}")

    print(f"[info] {len(chunks)} chunks détectés → {chunks[0]} ... {chunks[-1]}")

    model = TinyNNUE(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_seen = 0
        print(f"\n[Époque {epoch}/{EPOCHS}]")

        for chunk_id, chunk_path in enumerate(chunks):
            data = torch.load(chunk_path, weights_only=False)
            ds = SparseChunkDataset(data["idx"], data["y"], INPUT_DIM)
            loader = DataLoader(ds,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=0,       # Important pour Windows
                                pin_memory=True)

            pbar = tqdm(loader,
                        desc=f"[{epoch}/{EPOCHS}] Chunk {chunk_id+1}/{len(chunks)}",
                        unit="batch",
                        leave=False)

            for xb, yb in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * xb.size(0)
                total_seen += xb.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            del data, ds, loader
            torch.cuda.empty_cache()

        train_loss = total_train_loss / max(1, total_seen)
        print(f"[Époque {epoch}] train_loss={train_loss:.4f}")

        # Validation approximative = dernier chunk
        model.eval()
        val_data = torch.load(chunks[-1], weights_only=False)
        val_ds = SparseChunkDataset(val_data["idx"], val_data["y"], INPUT_DIM)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loss_total = 0.0
        val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_total += loss_fn(pred, yb).item() * xb.size(0)
                val_seen += xb.size(0)
        val_loss = val_loss_total / max(1, val_seen)
        print(f"→ val_loss (approx)={val_loss:.4f}")

        # Sauvegardes
        path_epoch = os.path.join(CHECKPOINT_DIR, f"nnue_stockfish_epoch{epoch}.pt")
        torch.save(model.state_dict(), path_epoch)
        print(f"[save] Modèle sauvegardé → {path_epoch}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "nnue_stockfish_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"[best] Nouveau meilleur modèle enregistré (val_loss={val_loss:.4f})")

    print("[done] Entraînement terminé.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
