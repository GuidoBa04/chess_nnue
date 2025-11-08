# eval_vs_stockfish.py
# Matchs NNUE vs Stockfish (faible) pour estimer un Elo relatif.
# Prérequis: python-chess, torch, pygame non requis.

import argparse
import math
import random
import time
from pathlib import Path

import chess
import chess.pgn
import chess.engine
import torch

from nnue_core import pick_device, load_model, search_best_move
from halfkp_encoder import parse_fen_or_startpos

# -------------------------------
# Helpers
# -------------------------------

def elo_from_score(p):
    # p = score moyen par partie (1=win, 0.5=draw, 0=loss)
    # ΔElo = 400 * log10(p/(1-p))
    if p <= 0.0: return -float("inf")
    if p >= 1.0: return float("inf")
    return 400.0 * math.log10(p / (1.0 - p))

def setup_stockfish(sf_path, skill=None, uci_elo=None, threads=1, hash_mb=32):
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    opts = {"Threads": threads, "Hash": hash_mb}
    # Deux façons de “faiblir” Stockfish:
    # 1) Skill Level (0..20)
    if skill is not None:
        opts["Skill Level"] = int(skill)
    # 2) UCI_LimitStrength + UCI_Elo (si binaire supporte cette option)
    if uci_elo is not None:
        opts["UCI_LimitStrength"] = True
        opts["UCI_Elo"] = int(uci_elo)
    engine.configure(opts)
    return engine

def pick_sf_limit(mode, value_ms_or_depth):
    if mode == "movetime":
        return chess.engine.Limit(time=value_ms_or_depth / 1000.0)
    elif mode == "depth":
        return chess.engine.Limit(depth=int(value_ms_or_depth))
    else:
        raise ValueError("mode doit être 'movetime' ou 'depth'")

def play_one_game(model, device, sf_engine, sf_limit, our_depth, our_color_white=True, pgn_out=None):
    board = parse_fen_or_startpos("startpos")
    game = chess.pgn.Game()
    game.headers["White"] = "NNUE" if our_color_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if our_color_white else "NNUE"
    node = game

    while not board.is_game_over():
        if (board.turn == chess.WHITE) == our_color_white:
            # Notre coup
            mv, score = search_best_move(board, our_depth, model, device)
            if mv is None:
                break
            board.push(mv)
            node = node.add_variation(mv)
        else:
            # Coup Stockfish
            try:
                res = sf_engine.play(board, sf_limit)
                mv = res.move
            except chess.engine.EngineTerminatedError:
                break
            except chess.engine.EngineError:
                break
            if mv is None:
                break
            board.push(mv)
            node = node.add_variation(mv)

    result = board.result(claim_draw=True)  # "1-0","0-1","1/2-1/2"
    if pgn_out is not None:
        game.headers["Result"] = result
        with open(pgn_out, "a", encoding="utf-8") as f:
            print(game, file=f, end="\n\n")
    if result == "1-0":
        return 1.0 if our_color_white else 0.0
    elif result == "0-1":
        return 0.0 if our_color_white else 1.0
    else:
        return 0.5

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Évaluer NNUE vs Stockfish faible")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint NNUE .pt")
    ap.add_argument("--stockfish", type=str, required=True, help="Chemin vers l'exécutable Stockfish")
    ap.add_argument("--games", type=int, default=40, help="Nombre total de parties")
    ap.add_argument("--our_depth", type=int, default=4, help="Profondeur de notre moteur")
    # Choix de l’affaiblissement Stockfish
    ap.add_argument("--sf_skill", type=int, default=None, help="Stockfish Skill Level (0..20). Ex: 1≈300, 3≈500, 5≈700 (approx)")
    ap.add_argument("--sf_elo", type=int, default=None, help="Stockfish UCI_Elo si supporté. Ex: 700, 900, 1100")
    # Temps/Profondeur SF
    ap.add_argument("--sf_mode", choices=["movetime", "depth"], default="movetime", help="Contrôle SF: 'movetime' ou 'depth'")
    ap.add_argument("--sf_value", type=int, default=200, help="ms si movetime, sinon depth si mode=depth")
    # Technique
    ap.add_argument("--threads", type=int, default=1, help="Threads SF")
    ap.add_argument("--hash", type=int, default=32, help="Hash MB SF")
    ap.add_argument("--pgn", type=str, default=None, help="Fichier PGN de sortie (append)")
    ap.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    args = ap.parse_args()

    random.seed(args.seed)

    # Modèle
    device = pick_device("auto")
    model = load_model(args.ckpt, device)

    # Stockfish
    sf_path = Path(args.stockfish)
    if not sf_path.exists():
        raise FileNotFoundError(f"Stockfish introuvable: {sf_path}")
    engine = setup_stockfish(str(sf_path), skill=args.sf_skill, uci_elo=args.sf_elo,
                             threads=args.threads, hash_mb=args.hash)
    sf_limit = pick_sf_limit(args.sf_mode, args.sf_value)

    # Matches
    wins = draws = losses = 0
    total_score = 0.0
    start = time.time()

    try:
        for i in range(args.games):
            our_white = (i % 2 == 1)
            pgn_out = args.pgn
            score = play_one_game(model, device, engine, sf_limit, args.our_depth, our_color_white=our_white, pgn_out=pgn_out)
            total_score += score
            if score == 1.0: wins += 1
            elif score == 0.5: draws += 1
            else: losses += 1
            print(f"[game {i+1}/{args.games}] result={'W' if score==1.0 else 'D' if score==0.5 else 'L'} | cum_score={total_score:.1f}")

    finally:
        engine.quit()

    elapsed = time.time() - start
    p = total_score / args.games  # score moyen
    delta_elo = elo_from_score(p)

    print("\n=== Résultats ===")
    print(f"Parties       : {args.games}")
    print(f"W/D/L         : {wins}/{draws}/{losses}")
    print(f"Score moyen   : {p:.3f}")
    print(f"ΔElo (relatif): {delta_elo:+.0f} (NNUE vs SF-config)")
    print(f"Durée         : {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()
