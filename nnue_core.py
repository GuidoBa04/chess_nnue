# nnue_core.py
# Noyau NNUE amélioré : modèle, chargement, évaluation et recherche (TT + ID + qsearch)


import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import chess
import time


from halfkp_encoder import halfkp_dense, INPUT_DIM



TIME_LIMIT=5.0  # secondes
MVV = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
       chess.ROOK:5, chess.QUEEN:9, chess.KING:100}

# Valeurs pour le MATERIEL pur (roi = 0)
MATERIAL_VALS = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900
}

# Valeurs pour SEE / QSearch (roi très grand pour couper net)
SEE_VALS = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10_000
}

CENTER_SQS = [chess.D4, chess.E4, chess.D5, chess.E5]

NULL_REDUCTION = 2  # réduction de profondeur pour Null Move Pruning
# ============================================================
# 1. Réseau NNUE
# ============================================================
class TinyNNUE(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x): return self.net(x)


# ============================================================
# 2. Chargement / périphérique
# ============================================================
def pick_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def load_model(ckpt_path: Optional[str], device: torch.device) -> TinyNNUE:
    model = TinyNNUE(INPUT_DIM).to(device)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            print(f"[info] Modèle chargé depuis: {ckpt_path}")
        except Exception as e:
            print(f"[warn] Échec du chargement '{ckpt_path}' ({e}) → poids aléatoires.")
    else:
        print("[info] Aucun checkpoint fourni → poids aléatoires utilisés.")
    model.eval()
    return model


# ============================================================
# 3. Évaluation
# ============================================================
@torch.inference_mode()
def eval_white_cp(board, model, device, detailed=False) -> float:
    x = halfkp_dense(board).unsqueeze(0).to(device)
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0

    
    nn_raw = float(model(x).item()) * 600.0
    nn_eval = nn_raw if board.turn == chess.WHITE else -nn_raw

    material_white = sum(MATERIAL_VALS[p.piece_type] for p in board.piece_map().values()
                     if p.color == chess.WHITE and p.piece_type in MATERIAL_VALS)
    material_black = sum(MATERIAL_VALS[p.piece_type] for p in board.piece_map().values()
                     if p.color == chess.BLACK and p.piece_type in MATERIAL_VALS)
    material = material_white - material_black  # >0 si avantage matériel blanc

    open_cp = opening_heuristics_side_cp(board, chess.WHITE) - opening_heuristics_side_cp(board, chess.BLACK)

    # --- pondération dynamique selon la phase de jeu ---
    # 1. Calcule la somme du matériel absolu (hors rois)
    total_material = material_white + material_black
    # 2. Estime la "phase d’ouverture" (1 = début, 0 = finale)
    opening_phase = min(1.0, total_material / 7800.0)  # ~7800 pour matériel initial
    # 3. Mélange : principes dominants en début, négligeables en fin
    w_nn = 0.5 + 0.3 * (1 - opening_phase)
    w_mat = 0.2
    w_open = 0.3 * opening_phase
    
    score = w_nn * nn_eval + w_mat * material + w_open * open_cp
    
    return (score, w_nn, nn_eval, w_mat, material, w_open, open_cp) if detailed else score

    

    

def terminal_eval_cp(board: chess.Board) -> Optional[float]:
    if not board.is_game_over():
        return None
    res = board.result(claim_draw=True)
    if res == "1-0":
        return +10_000.0
    elif res == "0-1":
        return -10_000.0
    return 0.0


# ============================================================
# 4. Transposition Table
# ============================================================
class TTEntry:
    __slots__ = ("depth", "score", "flag", "move")
    def __init__(self, depth: int, score: float, flag: str, move: Optional[chess.Move]):
        self.depth, self.score, self.flag, self.move = depth, score, flag, move

TT: Dict[int, TTEntry] = {}


# ============================================================
# 5. Heuristiques de tri de coups
# ============================================================



def gives_check(board, mv):
    board.push(mv)
    chk = board.is_check()
    board.pop()
    return chk

def order_moves(board, tt_move=None):
    """Trie les coups pour accélérer la recherche."""
    moves = list(board.legal_moves)

    # --- Coup TT d'abord ---
    if tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)

    scored = []
    for mv in moves:
        score = 0

        # Bonus TT
        if mv == tt_move:
            score += 20000

        # Captures : MVV-LVA + SEE pruning léger
        if board.is_capture(mv):
            v = board.piece_at(mv.to_square) #victime
            a = board.piece_at(mv.from_square) #attaquant
            if v and a:
                value_v = MVV.get(v.piece_type, 0)
                value_a = MVV.get(a.piece_type, 0)
                see = value_v - value_a
                score += 1000 + 10 * value_v - value_a #le but est de prednre une piece de forte valeur avec une piece de faible valeur
                if see < -1:  # capture perdante → fort malus
                    score -= 500

        # Coup donnant échec
        if gives_check(board, mv):
            score += 10

        # Coup de promotion
        if mv.promotion:
            score += 900

        scored.append((score, mv))

    # Tri descendant (meilleurs scores d'abord)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mv for _, mv in scored]






def is_castled_kingside(board, color):
    return board.king(color) in (chess.G1 if color==chess.WHITE else chess.G8,)

def is_castled_queenside(board, color):
    return board.king(color) in (chess.C1 if color==chess.WHITE else chess.C8,)

def pieces_heavy(board):
    # "beaucoup de pièces sur l’échiquier"
    return sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING) >= 20

def has_rook_connection(board, color):
    """Retourne True si les deux tours de départ sont connectées (aucune pièce entre elles sur la 1ère rangée)."""
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return False

    # trie par colonne
    files = sorted([chess.square_file(sq) for sq in rooks])
    rank = 0 if color == chess.WHITE else 7

    # vérifie qu’il n’y a rien entre la 1re et la dernière tour
    for f in range(files[0] + 1, files[-1]):
        if board.piece_at(chess.square(f, rank)) is not None:
            return False
    return True


def minor_on_backrank(board, color):
    rank = 0 if color==chess.WHITE else 7
    # mineures encore sur la 1re/8e rangée
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) == rank:
                return True
    return False

def knights_on_rim(board, color):
    # a/b/g/h files
    rim_files = {0,1,6,7}
    return sum(1 for sq in board.pieces(chess.KNIGHT, color)
               if chess.square_file(sq) in rim_files)

def queen_developed_too_early(board, color):
    start = chess.D1 if color==chess.WHITE else chess.D8
    qsq = next(iter(board.pieces(chess.QUEEN, color)), None)
    if qsq is None:
        return False
    # trop tôt = dame sortie alors que mineures sont à la maison et peu de coups joués
    early = board.fullmove_number <= 8
    return early and qsq != start and minor_on_backrank(board, color)

def pawn_on(board, color, file_char, rank_num):
    file_idx = ord(file_char)-ord('a')
    rank_idx = rank_num-1
    sq = chess.square(file_idx, rank_idx)
    p = board.piece_at(sq)
    return p is not None and p.color==color and p.piece_type==chess.PAWN

def pawn_moved_from_start(board, color, file_char):
    # f2/f7, g2/g7 ont-ils bougé ?
    start_sq = chess.square(ord(file_char)-97, 1 if color==chess.WHITE else 6)
    p = board.piece_at(start_sq)
    return not (p is not None and p.piece_type==chess.PAWN and p.color==color)

def opening_heuristics_side_cp(board, color) -> int:
    cp = 0

    # 1) Roque
    castled_s = is_castled_kingside(board, color) or is_castled_queenside(board, color)
    if castled_s:
        cp += 200  # bonus roque
        # évite d’ouvrir f/g après roque petit
        if is_castled_kingside(board, color):
            if pawn_moved_from_start(board, color, 'f'): cp -= 30
            if pawn_moved_from_start(board, color, 'g'): cp -= 30
    else:
        # roi a déjà bougé → plus de droits de roque et roi hors e1/e8
        king_sq = board.king(color)
        start_king = chess.E1 if color==chess.WHITE else chess.E8
        still_can_castle = (board.has_kingside_castling_rights(color) or
                            board.has_queenside_castling_rights(color))
        if king_sq != start_king and not still_can_castle and pieces_heavy(board):
            cp -= 100  # a bougé avant de roquer alors que la position est encore lourde
        # retard de roque passé ~12e coup
        if board.fullmove_number >= 12 and pieces_heavy(board):
            cp -= 40

    # 2) Développement mineures
    rank0 = 0 if color==chess.WHITE else 7
    undeveloped = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) == rank0:
                undeveloped += 1
    cp += 15 * (2 - min(2, undeveloped))  # bonus si sorties
    if board.fullmove_number >= 8 and undeveloped > 0:
        cp -= 20 * undeveloped  # retard de développement

    # 3) Centre
    # pions au centre
    targets = [chess.D4, chess.E4] if color==chess.WHITE else [chess.D5, chess.E5]
    for sq in targets:
        p = board.piece_at(sq)
        if p and p.color==color and p.piece_type==chess.PAWN:
            cp += 20
    # cases du centre attaquées
    attacks = 0
    for sq in CENTER_SQS:
        if board.is_attacked_by(color, sq):
            attacks += 1
    cp += 5 * attacks

    # 4) Tours connectées
    if has_rook_connection(board, color):
        cp += 20

    # 5) Cavaliers sur la bande
    cp -= 10 * knights_on_rim(board, color)

    # 6) Dame trop tôt
    if queen_developed_too_early(board, color):
        cp -= 25

    return cp

# ============================================================
# 6. Quiescence Search
# ============================================================



def see_gain(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0
    v = board.piece_at(move.to_square)
    a = board.piece_at(move.from_square)
    if v and a:
        return SEE_VALS[v.piece_type] - SEE_VALS[a.piece_type]
    return 0

@torch.inference_mode()
def qsearch(board, alpha, beta, model, device, last_move=None):
    
    stand = eval_white_cp(board, model, device)
    stand = stand if board.turn == chess.WHITE else -stand
    if stand >= beta: return beta
    if stand > alpha: alpha = stand

    caps = [m for m in board.legal_moves if board.is_capture(m)]
    if last_move:
        caps.sort(key=lambda m: (m.to_square == last_move.to_square,
                                 board.is_capture(m)), reverse=True)
    else:
        caps.sort(key=lambda m: (board.is_capture(m)), reverse=True)

    for mv in caps:
        if see_gain(board, mv) < -50 and board.piece_at(mv.to_square).piece_type not in (chess.QUEEN, chess.KING):
            continue
        board.push(mv)
        score = -qsearch(board, -beta, -alpha, model, device, last_move=mv)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


# ============================================================
# 7. Negamax avec TT, Null-Move et Quiescence
# ============================================================


def negamax(board, depth, alpha, beta, model, device, last_move=None, ply=0):
    key = board._transposition_key()
    entry = TT.get(key)
    if entry and entry.depth >= depth:
        if entry.flag == "EXACT":
            return entry.score
        if entry.flag == "LOWER" and entry.score > alpha:
            alpha = entry.score
        elif entry.flag == "UPPER" and entry.score < beta:
            beta = entry.score
        if alpha >= beta:
            return entry.score

    term = terminal_eval_cp(board)
    if term is not None:
        return term if board.turn == chess.WHITE else -term

    # --- profondeur nulle → Quiescence ---
    if depth == 0:
        return qsearch(board, alpha, beta, model, device, last_move=last_move)

    # --- Null Move Pruning ---
    if depth >= 3 and not board.is_check() and len(list(board.legal_moves)) > 0:
        board.push(chess.Move.null())
        score = -negamax(board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1,
                         model, device, last_move=None, ply=ply + 1)
        board.pop()
        if score >= beta:
            return beta

    best_score = -float("inf")
    best_move = None
    tt_move = entry.move if entry else None

    # --- Boucle sur les coups ---
    for i, mv in enumerate(order_moves(board, tt_move)):
        # ---- Extensions ----
        ext = 0
        if board.is_check():  # allonger en cas d’échec
            ext += 1
        if last_move and mv.to_square == last_move.to_square:  # recapture
            ext += 1
        if mv.promotion:  # promotion
            ext += 1
        new_depth = max(0, depth - 1 + ext)

        # ---- Late Move Reduction ----
        reduce = 0
        if ext == 0 and i >= 3 and not board.is_capture(mv) \
           and not gives_check(board, mv) and depth >= 3:
            reduce = 1

        board.push(mv)
        score = -negamax(board, new_depth - reduce, -beta, -alpha,
                         model, device, last_move=mv, ply=ply + 1)
        board.pop()

        # ---- Mise à jour ----
        if score > best_score:
            best_score, best_move = score, mv
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    # --- Mémorisation TT ---
    flag = "EXACT"
    if best_score <= alpha:
        flag = "UPPER"
    elif best_score >= beta:
        flag = "LOWER"
    TT[key] = TTEntry(depth, best_score, flag, best_move)
    return best_score



# ============================================================
# 8. Iterative Deepening + affichage
# ============================================================
def _format_score(score):
    if abs(score) > 9000:
        m = int((10000 - abs(score)) / 100)
        return f"M{m}" if score > 0 else f"M-{m}"
    return f"{score:+.1f} cp"

def _search_depth(board, depth, model, device, start_time, time_limit):

    alpha, beta = -float("inf"), float("inf")
    best_move, best_score = None, -float("inf")
    tt_entry = TT.get(board._transposition_key())
    tt_move = tt_entry.move if tt_entry else None

    for mv in order_moves(board, tt_move):
        if start_time and time_limit and time.time() - start_time > time_limit:
            print(f"[ID] Limite de temps atteinte dans la profondeur {depth}.")
            break
        board.push(mv)
        score = -negamax(board, depth - 1, -beta, -alpha, model, device)
        board.pop()
        if score > best_score:
            best_score, best_move = score, mv
        if best_score > alpha:
            alpha = best_score
    return best_move, best_score


def search_best_move(board, depth, model, device, show_components=True, time_limit=TIME_LIMIT) :
    if board.is_game_over():
        return None, 0.0
    
    start_time = time.time()

    best_move, best_score = None, 0.0
    for d in range(1, depth + 1):
        if time.time() - start_time > time_limit:
            print(f"[ID] Limite de temps atteinte après {d-1} itérations.")
            break
        mv, sc = _search_depth(board, d, model, device, start_time, time_limit)
        if mv is not None:
            best_move, best_score = mv, sc
            print(f"[ID] depth {d}: {mv.uci()} ({_format_score(sc)})")
            if show_components:
                tmp = board.copy()
                tmp.push(mv)
                s, w_nn, nn_c, w_mat, mat_c, w_open, open_c = eval_white_cp(tmp, model, device, detailed=True)
                # Les composantes retournées sont en “côté Blancs”. Pour une lecture “côté au trait de tmp”:
                if tmp.turn == chess.BLACK:
                    nn_c, mat_c, open_c = -nn_c, -mat_c, -open_c
                print(f"     ↳ NN={nn_c:+.1f}  Mat={mat_c:+.1f}  Open={open_c:+.1f}")

        else:
            print(f"[ID] depth {d}: (aucun coup possible)")
    return best_move, best_score

