# check_classes.py

import numpy as np
import os

# Caminho para o arquivo que armazena as classes
LABELS_PATH = "label_encoder_classes.npy"

# Verifica se o arquivo existe
if os.path.exists(LABELS_PATH):
    print(f"Lendo as classes do arquivo: '{LABELS_PATH}'")
    
    # Carrega o array de classes
    known_classes = np.load(LABELS_PATH, allow_pickle=True)
    
    print("\nO seu modelo foi treinado para reconhecer as seguintes classes (letras):")
    # Imprime as classes em formato de lista
    print(list(known_classes))
    
    print(f"\nTotal de classes: {len(known_classes)}")
    
else:
    print(f"ERRO: O arquivo '{LABELS_PATH}' não foi encontrado.")
    print("Você precisa rodar o script 'trainingmodel.py' primeiro para gerar o modelo e o arquivo de classes.")