# ♟️ NNUE Chess Engine

Moteur d'échecs implémenté en Python, combinant un réseau **NNUE** (Efficiently Updatable Neural Network) avec une **recherche Negamax optimisée** (alpha–bêta, quiescence, LMR, etc.).  
Le projet permet d'entraîner le réseau sur des données de Stockfish et de jouer contre le moteur via une interface graphique.

---

## 🚀 Fonctionnalités principales

- Réseau de neurones **TinyNNUE** (évaluation efficace en centipions)
- **Recherche Negamax** avec :
  - Élagage **α–β**
  - **Table de transposition**
  - **Null move pruning**
  - **Late move reductions (LMR)**
  - **Quiescence search**
- Pondération dynamique entre évaluation NNUE, matériel et principes d'ouverture
- Entraînement supervisé à partir de positions évaluées par Stockfish
- Interface **Pygame** pour jouer contre l'IA
- Script d'évaluation automatique contre Stockfish à différents niveaux

---

## 📦 Installation
```bash
git clone https://github.com/GuidoBa04/chess_nnue.git
cd chess_nnue
```
```bash
pip install -r requirements.txt
```

Vérifier que le fichier `dataset_stockfish.csv` est présent à la racine du projet. Il contient les positions FEN et les évaluations Stockfish utilisées pour l'entraînement.

Sinon, vous pouvez le télécharger ici : https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations?resource=download

---

## 🎓 Entraînement du modèle

### 1️⃣ Encodage des positions

Les positions du dataset sont transformées en représentation HalfKP sparse :

```bash
python encode_halfkp_sparse.py
```

Ce script utilise `halfkp_encoder.py` pour encoder les positions et créer des chunks d'entraînement enregistrés dans le dossier `encoded_sparse_chunks/`.

### 2️⃣ Entraînement du modèle NNUE

On lance l'apprentissage supervisé du réseau à partir des chunks encodés :

```bash
python train_stockfish_chunks.py
```

Le script sauvegarde les modèles intermédiaires et le meilleur modèle dans le dossier `checkpoints/` sous forme de fichiers `.pt` (PyTorch).

---

## 🧩 Utilisation du moteur

### 1️⃣ Évaluation et fonctions internes

Le fichier `nnue_core.py` contient le cœur du moteur :

- Définition du modèle `TinyNNUE`
- Fonctions d'évaluation (`eval_white_cp`)
- Algorithme negamax avec TT et pruning
- Recherche du meilleur coup (`search_best_move`)

Ces fonctions peuvent être réutilisées dans d'autres projets ou pour l'analyse de positions.

### 2️⃣ Jouer contre l'ordinateur (interface graphique)

Lancer l'interface Pygame pour affronter le moteur NNUE :

```bash
python gui_chess_vs_engine.py
```

**Commandes principales :**

- Cliquez sur une pièce puis sur la case de destination pour jouer un coup
- Le moteur joue ensuite automatiquement son coup (profondeur par défaut : 3)
- Le plateau se met à jour en temps réel

### 3️⃣ Évaluer le moteur face à Stockfish

Pour comparer les performances du moteur NNUE à différentes profondeurs ou niveaux de Stockfish  :

```bash
python eval_vs_stockfish.py \
  --ckpt checkpoints/nnue_stockfish_best.pt \
  --stockfish "C:\Users\bapti\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe" \
  --games 1 \
  --our_depth 3 \
  --sf_skill 1 \
  --sf_mode movetime --sf_value 200 \
  --pgn results_sf_skill1.pgn
```

| Argument      | Description |
|----------------|-------------|
| `--ckpt`       | Chemin vers le modèle NNUE entraîné (`.pt`) |
| `--stockfish`  | Chemin vers l’exécutable Stockfish |
| `--games`      | Nombre de parties à jouer |
| `--our_depth`  | Profondeur de recherche du moteur NNUE entraîné |
| `--sf_skill`   | Niveau de Stockfish (0–20) |
| `--sf_mode`    | Mode de limitation (`movetime` ou `depth`) |
| `--sf_value`   | Valeur associée au mode choisi (ex. `200` ms ou profondeur) |
| `--pgn`        | Nom du fichier PGN où seront enregistrées les parties |



## 📊 Exemple de sortie console

```
[info] Modèle chargé depuis: checkpoints/nnue_stockfish_best.pt
[ID] depth 1: d7d6 (-5.9 cp)
     ↳ NN=-0.2  Mat=+0.0  Open=+20.0
[ID] depth 2: e7e5 (-8.9 cp)
     ↳ NN=+33.7  Mat=+0.0  Open=+0.0
[ID] depth 3: e7e5 (-7.7 cp)
     ↳ NN=+33.7  Mat=+0.0  Open=+0.0
```

---

## 🧱 Structure du projet

```
chess_nnue/
│
├── dataset_stockfish.csv        # Données d'entraînement
├── halfkp_encoder.py            # Encodage HalfKP (pièces + cases)
├── encode_halfkp_sparse.py      # Création des chunks à partir du dataset
├── train_stockfish_chunks.py    # Entraînement du modèle NNUE
├── nnue_core.py                 # Cœur du moteur (évaluation et recherche)
├── gui_chess_vs_engine.py       # Interface graphique Pygame
├── eval_vs_stockfish.py         # Évaluation automatique vs Stockfish
├── checkpoints/                 # Dossier contenant les modèles .pt
├── encoded_sparse_chunks/       # Données d'entraînement encodées
└── results_sf_skill1.pgn        # Résultats des parties contre Stockfish
```

---

## 🧩 Théorie rapide

Le moteur combine :

- Un réseau NNUE évaluant les positions via un encodage HalfKP efficace
- Une recherche Negamax optimisée (α–β, TT, LMR, Null Move)
- Une pondération dynamique entre réseau, matériel et principes d'ouverture
- Un approfondissement itératif pour choisir le meilleur coup

---

## 📈 Perspectives d'amélioration

- Entraînement sur un dataset plus large (plus de parties Stockfish)
- Ajout de certaines variables d'entraînement (variance) et évaluation du modèle suivant la phase de jeu (ouverture, milieu de jeu, finale, tactique, mat en x coups, etc.)
- Ajout de tablebases pour les finales (lorsque moins de 7 pièces sont présentes sur l'échiquier)
- Ajout d'un programme d'ouverture
- Optimisation du temps de recherche via C++ ou CUDA
- Ammélioration de l'algorithme de choix des coups
- Ajout d'un réel moteur d'évaluation du classement élo