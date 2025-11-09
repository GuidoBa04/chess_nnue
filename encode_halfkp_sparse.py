# encode_halfkp_sparse.py
# Stocke uniquement les indices HalfKP actifs par position
# Gain ~x800 sur l'espace disque

import os
import torch
import pandas as pd
from tqdm import tqdm
from halfkp_encoder import halfkp_active_indices, parse_fen_or_startpos

CSV_PATH = "dataset_stockfish.csv"
OUT_DIR = "encoded_sparse_chunks"
CHUNK_SIZE = 10_000
MAX_CP = 600
LIMIT = None   # None pour tout le dataset, mettre un entier N pour limiter à N positions

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[info] Lecture du dataset : {CSV_PATH}")
df_iter = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
total = 0

for chunk_id, df in enumerate(df_iter):
    if LIMIT and total >= LIMIT:
        break
    if LIMIT:
        df = df.head(max(0, LIMIT - total))

    n = len(df)
    print(f"[chunk {chunk_id}] Encodage sparse de {n} positions...")

    sparse_data = []
    evals = []

    for row in tqdm(df.itertuples(index=False), total=n, desc=f"Chunk {chunk_id:04d}"):
        fen = row.FEN
        e = str(row.Evaluation).strip()

        # Nettoyage évaluation
        if e.startswith("#"):
            eval_cp = 10000.0
        elif e.startswith("-#"):
            eval_cp = -10000.0
        else:
            e = e.replace("+", "")
            try:
                eval_cp = float(e)
            except ValueError:
                eval_cp = 0.0
        eval_cp = max(-MAX_CP, min(MAX_CP, eval_cp))
        norm_eval = eval_cp / MAX_CP

        # Encode indices actifs
        board = parse_fen_or_startpos(fen)
        idx = torch.tensor(halfkp_active_indices(board), dtype=torch.int32)  # typiquement <512 indices
        sparse_data.append(idx)
        evals.append(norm_eval)

    out_path = os.path.join(OUT_DIR, f"encoded_sparse_{chunk_id:04d}.pt")
    torch.save({"idx": sparse_data, "y": torch.tensor(evals, dtype=torch.float16)}, out_path)
    print(f"[OK] {n:,} positions → {out_path}")
    total += n

print(f"[done] {total:,} positions encodées (sparse).")
