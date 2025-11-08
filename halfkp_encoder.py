# halfkp_encoder.py
# Étape 1 : encodeur demi-KP (HalfKP) + smoke test minimal
# Dépendances: python-chess, torch
#   pip install python-chess torch

import chess
import torch
from typing import List, Tuple

# Mapping des types de pièces non-roi -> index [0..4]
PIECE_TO_HALF = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
}

NUM_PERSPECTIVES = 2     # 0 = side-to-move : le camp qui a le trait , 1 = le camp qui n'a pas le trait
NUM_KING_SQ = 64 #le nbr de case pour le roi de la perspective
NUM_TYPES = 5            # P,N,B,R,Q (pas de roi)
NUM_SQ = 64 #nbr de cases possibles pr les pieces non-roi
INPUT_DIM = NUM_PERSPECTIVES * NUM_KING_SQ * NUM_TYPES * NUM_SQ  # 40_960

def flip_vertical(square: int) -> int:
    """Retourne la case en inversant le rang (a1<->a8, ...). car on regarde le board depuis la perspective de celui qui a le trait"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return chess.square(file, 7 - rank)

def sq_for_perspective(sq: int, perspective_is_white: bool) -> int:
    """Convertit une case dans la perspective (blanc = non-flip, noir = flip vertical)."""
    return sq if perspective_is_white else flip_vertical(sq)

def king_sq_for(board: chess.Board, perspective_is_white: bool) -> int:
    """Case du roi du camp de la perspective (après éventuel flip)."""
    ksq = board.king(chess.WHITE if perspective_is_white else chess.BLACK)
    if ksq is None:
        raise ValueError("Roi manquant dans la position (FEN invalide ?)")
    return sq_for_perspective(ksq, perspective_is_white)

def iter_non_king_pieces(board: chess.Board):
    """Itère (color, ptype, square) pour toutes les pièces non-roi présentes."""
    for color in (chess.WHITE, chess.BLACK):
        for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(ptype, color):
                yield color, ptype, sq #yield transforme la fonction en générateur : au lieu de retourner une liste complète d’un coup, on “produit” les éléments un 
                                        #par un à la demande (for ... in iter_non_king_pieces(board)).
                                        #Avantages : mémoire (pas de grosse liste temporaire) et simplicité pour itérer.

def halfkp_active_indices(board: chess.Board) -> List[int]:
    """
    Retourne la liste d'indices HalfKP actifs (sparse) pour une position.
    Dimensions globales: 40_960.
    Indexation:
      base_p      = p_idx * (64*5*64)
      block_king  = king_sq_p * (5*64)
      block_type  = type_idx * 64
      index       = base_p + block_king + block_type + piece_sq_p
    """
    indices: List[int] = []

    stm_is_white = board.turn
    # p_idx=0 => perspective side-to-move (au trait), p_idx=1 => other
    perspectives: List[Tuple[int, bool]] = [(0, stm_is_white), (1, not stm_is_white)]

    for p_idx, persp_is_white in perspectives:
        base_p = p_idx * (NUM_KING_SQ * NUM_TYPES * NUM_SQ)
        ksq_p = king_sq_for(board, persp_is_white)           # roi de la perspective
        block_king = ksq_p * (NUM_TYPES * NUM_SQ)

        for color, ptype, sq in iter_non_king_pieces(board):
            type_idx = PIECE_TO_HALF[ptype]
            piece_sq_p = sq_for_perspective(sq, persp_is_white)
            idx = base_p + block_king + (type_idx * NUM_SQ) + piece_sq_p
            indices.append(idx)

    return indices

def halfkp_dense(board: chess.Board) -> torch.Tensor:
    """Retourne le vecteur d'entrée dense (40_960,) en float32 avec 0/1."""
    idx = halfkp_active_indices(board)
    x = torch.zeros(INPUT_DIM, dtype=torch.float32)
    # S'il y a dupli (très improbable), on clippe à 1.0 de toute façon.
    x[idx] = 1.0
    return x

# ---------- Smoke test minimal ----------
TOY = [
    # (FEN, eval_cp) - eval fictive juste pour vérifier la forme des batchs
    ("startpos", 0),  # alias: position initiale
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),
    # Milieu de jeu simple
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", 35),
    # Un échange effectué
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 3 3", -20),
]

def parse_fen_or_startpos(fen: str) -> chess.Board: #si on a "startpos" on retourne la position initiale
    if fen == "startpos":
        return chess.Board()
    return chess.Board(fen)

def demo():
    print("== Demo HalfKP ==")
    for i, (fen, eval_cp) in enumerate(TOY):
        board = parse_fen_or_startpos(fen)
        x = halfkp_dense(board)
        active = int(x.sum().item())
        # Nombre de features actives = (# pièces non-roi) * 2 perspectives
        num_non_king = sum(len(list(board.pieces(pt, chess.WHITE))) +
                           len(list(board.pieces(pt, chess.BLACK)))
                           for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        expected_active = num_non_king * 2

        print(f"[{i}] FEN: {fen[:50]}{'...' if len(fen)>50 else ''}")
        print(f"    x.shape      = {tuple(x.shape)} (doit être (40960,))")
        print(f"    x.sum()      = {active}  (attendu ≈ {expected_active})")
        # Normalisation cible (ex: clamp +/-600 puis /600)
        target = max(-600, min(600, int(eval_cp))) / 600.0
        print(f"    target_norm  = {target:.3f}  (à prédire)")

if __name__ == "__main__":
    demo()
