# gui_chess_vs_engine.py
# Interface graphique Pygame : humain vs moteur NNUE

import pygame
import chess
import torch

from nnue_core import pick_device, load_model, search_best_move
from halfkp_encoder import parse_fen_or_startpos

# === Paramètres ===
CASE_SIZE = 80
BOARD_SIZE = 8
WINDOW_SIZE = CASE_SIZE * BOARD_SIZE
ICON_PATH = "icons" # chemin vers les icônes des pièces au format png
DEPTH = 3  # profondeur moteur

# === Couleurs ===
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)
TEXT_COLOR = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("NNUE Chess GUI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 36, bold=True)

# === Chargement icônes ===
def load_icons():
    icons = {}
    for color in ["white", "black"]:
        for name in ["P", "N", "B", "R", "Q", "K"]:
            path = f"{ICON_PATH}/{color}/{name}.png"
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.smoothscale(img, (CASE_SIZE, CASE_SIZE))
            icons[(color, name)] = img
    return icons

ICONS = load_icons()

# === Fonctions d'affichage ===
def draw_board(board, selected=None):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = LIGHT if (r + c) % 2 == 0 else DARK
            if selected is not None:
                sel_rank, sel_file = divmod(selected, 8)
                sel_rank = 7 - sel_rank
                if sel_rank == r and sel_file == c:
                    color = HIGHLIGHT
            pygame.draw.rect(screen, color, (c * CASE_SIZE, r * CASE_SIZE, CASE_SIZE, CASE_SIZE))

    for sq, piece in board.piece_map().items():
        rank = 7 - chess.square_rank(sq)
        file = chess.square_file(sq)
        color = "white" if piece.symbol().isupper() else "black"
        key = (color, piece.symbol().upper())
        screen.blit(ICONS[key], (file * CASE_SIZE, rank * CASE_SIZE))

def get_square_under_mouse(pos):
    x, y = pos
    file = x // CASE_SIZE
    rank = 7 - (y // CASE_SIZE)
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

# === Promotion ===
def choose_promotion(board, from_sq, to_sq):
    color = "white" if board.turn == chess.WHITE else "black"
    pieces = ["Q", "R", "B", "N"]
    rects = []
    menu_width = CASE_SIZE * 4
    menu_x = (WINDOW_SIZE - menu_width) // 2
    menu_y = (WINDOW_SIZE - CASE_SIZE) // 2
    screen.fill((50, 50, 50))
    for i, name in enumerate(pieces):
        img = ICONS[(color, name)]
        x = menu_x + i * CASE_SIZE
        y = menu_y
        screen.blit(img, (x, y))
        rects.append(pygame.Rect(x, y, CASE_SIZE, CASE_SIZE))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(rects):
                    if rect.collidepoint(event.pos):
                        mapping = {"Q": chess.QUEEN, "R": chess.ROOK,
                                   "B": chess.BISHOP, "N": chess.KNIGHT}
                        return mapping[pieces[i]]
        clock.tick(30)

# === Fin de partie ===
def show_game_over(board):
    msg = "Échec et mat" if board.is_checkmate() else \
          "Pat" if board.is_stalemate() else \
          "Nulle" if board.is_insufficient_material() else "Partie terminée"
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    text = font.render(msg, True, TEXT_COLOR)
    rect = text.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//2))
    screen.blit(text, rect)
    pygame.display.flip()
    wait = True
    while wait:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                wait = False
        clock.tick(15)

# === Boucle principale ===
def main():
    # Configuration moteur
    device = pick_device("auto")
    print(f"Using device: {device}")
    model = load_model("checkpoints/nnue_stockfish_best.pt", device) # chemin vers le modèle

    board = parse_fen_or_startpos("startpos")
    human_color = chess.WHITE
    selected = None
    running = True
    game_over = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif not game_over and board.turn == human_color and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sq = get_square_under_mouse(event.pos)
                if sq is None:
                    continue
                if selected is None:
                    if board.piece_at(sq) and board.piece_at(sq).color == board.turn:
                        selected = sq
                else:
                    move = chess.Move(selected, sq)
                    if move not in board.legal_moves:
                        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                            move_promo = chess.Move(selected, sq, promotion=promo)
                            if move_promo in board.legal_moves:
                                promo_choice = choose_promotion(board, selected, sq)
                                move = chess.Move(selected, sq, promotion=promo_choice)
                                break
                    if move in board.legal_moves:
                        board.push(move)
                    selected = None

        # Coup moteur
        
        if not game_over and board.turn != human_color:
            if not board.is_game_over():
                draw_board(board)
                pygame.display.flip()
                move, score = search_best_move(board, DEPTH, model, device)
                if move is not None:
                    board.push(move)


        draw_board(board, selected)
        pygame.display.flip()

        if not game_over and board.is_game_over():
            game_over = True
            show_game_over(board)

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
