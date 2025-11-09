# ♟️ NNUE Chess Engine

Moteur d'échecs implémenté en Python, combinant un réseau **NNUE** (Efficiently Updatable Neural Network) avec une **recherche Negamax optimisée** (alpha–bêta, quiescence, LMR, etc.).  
Le projet permet d'entraîner le réseau sur des données de Stockfish et de jouer contre le moteur via une interface graphique.

---

## Fonctionnalités principales

- **NNUE** :
Le réseau utilisé ici s’inspire de l’architecture **NNUE (Efficiently Updatable Neural Network Evaluation)** popularisée par Stockfish.  
Il s’agit d’un **réseau dense de petite taille** (TinyNNUE) implémenté en PyTorch, entraîné sur des positions évaluées par Stockfish.

Contrairement à un NNUE complet, cette version ne met pas encore à jour ses activations de façon incrémentale après chaque coup,  
mais conserve la même logique d’entrée (encodage HalfKP) et la même finalité :  
fournir une **évaluation rapide et apprise** des positions d’échecs. L'évaluation est donné en centipions, +100cp indique un avantage équivalent à un pion pour les blancs. 


- **Recherche Negamax** avec :
  - **Élagage α–β** : limite l’exploration de l’arbre aux coups nécessaires.  
    Dès qu’un coup ne peut pas améliorer le résultat final, la branche est coupée.
  - **Table de transposition** : table de hachage où sont stockées les positions déjà évaluées pour éviter les recalculs.
  - **Null move pruning** : simule un coup « nul » pour tester si la position est déjà suffisamment forte et ainsi gagner du temps de recherche.
  - **Late move reductions (LMR)** : réduit la profondeur d’analyse pour les coups moins prometteurs (généralement testés en fin de liste).
  - **Quiescence search** : prolonge la recherche lorsque la position est tactiquement instable (par ex. échanges de pièces en cours).

- **Pondération dynamique** entre :
  - l’évaluation NNUE (apprise),
  - le matériel (valeur des pièces),
  - et des heuristiques d’ouverture (développement, roque, contrôle du centre).

- **Entraînement supervisé** à partir de positions évaluées par Stockfish (le moteur de référence)

- **Interface Pygame** pour jouer contre l’IA localement.

- **Script d’évaluation automatique** (`eval_vs_stockfish.py`) pour comparer les performances du moteur NNUE à différents niveaux de Stockfish.

---

## L’encodage HalfKP (sparse) plutôt qu’une position brute

L’encodage **HalfKP** (Half-King–Piece) est utilisé à la place d’une représentation brute de la position (type matrice 8×8×12) pour des raisons d’efficacité, de rapidité et de compatibilité avec l’approche NNUE.

- **Principe :** chaque feature encode une paire *(roi, autre pièce)*.  
  Le réseau apprend donc les relations spatiales entre la position du roi et celle de chaque autre pièce sur l’échiquier.  
  Cela permet de **mettre à jour localement** les features après chaque coup, sans recalcul complet du réseau.

- **Efficacité mémoire et calcul :**  
  Une position brute (12×8×8 = 768 features denses) nécessiterait de recalculer toutes les activations après chaque coup.  
  L’encodage **HalfKP** ne met à jour que les éléments affectés, rendant l’évaluation **incrémentale et rapide**.  

- **Cohérence positionnelle :**  
  Centrer la représentation autour du roi permet au modèle de mieux capturer la **sécurité du roi**, les **liaisons pièces-roi**, et les motifs tactiques réalistes qui influencent fortement la valeur d’une position.

- **Format standard pour NNUE :**  
  La majorité des moteurs modernes (Stockfish, Lc0-NNUE, Berserk) utilisent HalfKP pour garantir des performances stables et une bonne généralisation.

- **Version sparse (HalfKP sparse) :**  
  Pour l’entraînement, on utilise une **représentation creuse** des features : seule une petite partie du vecteur d’entrée est non nulle pour chaque position.  
  Cela permet de **réduire drastiquement la taille des fichiers d’entraînement** et d’accélérer le chargement et le calcul lors de l’apprentissage avec PyTorch.  
  Les données sont prétraitées avec `encode_halfkp_sparse.py`, qui transforme `dataset_stockfish.csv` en **chunks encodés sparse** stockés dans `encoded_sparse_chunks/`.

--- 
## Installation
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
## Entraînement du modèle

### Encodage des positions

Les positions du dataset sont transformées en représentation HalfKP sparse :

```bash
python encode_halfkp_sparse.py
```

Ce script utilise `halfkp_encoder.py` pour encoder les positions et créer des chunks d'entraînement enregistrés dans le dossier `encoded_sparse_chunks/`.

### Entraînement du modèle NNUE

On lance l'apprentissage supervisé du réseau à partir des chunks encodés :

```bash
python train_stockfish_chunks.py
```

Le script sauvegarde les modèles intermédiaires et le meilleur modèle dans le dossier `checkpoints/` sous forme de fichiers `.pt` (PyTorch).



---

## Utilisation du moteur

### Évaluation et fonctions internes

Le fichier `nnue_core.py` contient le cœur du moteur :

- Définition du modèle `TinyNNUE`
- Fonctions d'évaluation (`eval_white_cp`)
- Algorithme negamax avec TT et pruning
- Recherche du meilleur coup (`search_best_move`)

Ces fonctions peuvent être réutilisées dans d'autres projets ou pour l'analyse de positions.

### Jouer contre l'ordinateur (interface graphique)

Lancer l'interface Pygame pour affronter le moteur NNUE :

```bash
python gui_chess_vs_engine.py
```

**Commandes principales :**

- Cliquez sur une pièce puis sur la case de destination pour jouer un coup
- Le moteur joue ensuite automatiquement son coup (profondeur par défaut : 3)
- Le plateau se met à jour en temps réel

### Évaluer le moteur face à Stockfish

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



## Exemple de sortie console

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

## Structure du projet

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
└── Icons/                       # Image png des pièces du jeu

```

---

## Perspectives d'amélioration

- Passage a un vrai réseau NNUE
- Entraînement sur un dataset plus large (plus de parties Stockfish)
- Ajout de certaines variables d'entraînement (variance) et évaluation du modèle suivant la phase de jeu (ouverture, milieu de jeu, finale, tactique, mat en x coups, etc.)
- Ajout de tablebases pour les finales (lorsque moins de 7 pièces sont présentes sur l'échiquier)
- Ajout d'un programme d'ouverture
- Ammélioration de l'algorithme de choix des coups
- Ajout d'un réel moteur d'évaluation du classement élo