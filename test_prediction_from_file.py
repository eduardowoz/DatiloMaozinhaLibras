import os

import numpy as np
import pandas as pd
from tensorflow import keras

# --- FUNÇÕES DE PREPARAÇÃO (CÓPIAS EXATAS DO trainingmodel.py) ---


def prepare_single_sequence(data, n_frames):
    """Prepara uma única amostra de 30 frames."""
    frames = data.loc[:, "x0":].values
    if len(frames) < n_frames:
        pad = np.zeros((n_frames - len(frames), frames.shape[1]))
        frames = np.vstack([frames, pad])
    else:
        frames = frames[:n_frames]
    return frames


def normalize_single_sample(X_sample):
    """Aplica a normalização (pulso + escala) em uma única amostra."""
    X_reshaped = X_sample.reshape(1, X_sample.shape[0], 21, 3)
    X_normalized = np.zeros_like(X_reshaped, dtype=float)
    for j in range(X_reshaped.shape[1]):
        frame_landmarks = X_reshaped[0, j]
        base_point = frame_landmarks[0].copy()
        normalized_frame = frame_landmarks - base_point
        max_value = np.max(np.linalg.norm(normalized_frame, axis=1))
        if max_value > 1e-6:
            normalized_frame /= max_value
        X_normalized[0, j] = normalized_frame
    return X_normalized


# --- CONFIGURAÇÕES ---
DATA_DIR = "dataset_libras"
MODEL_PATH = "modelo_todas_letras.keras"
LABELS_PATH = "label_encoder_classes.npy"
FRAMES_PER_SAMPLE = 30

# --- INÍCIO DO SCRIPT DE TESTE ---

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(
            "ERRO: Arquivos de modelo não encontrados. Rode 'trainingmodel.py' primeiro."
        )
        exit()

    print("--- Teste de Predição a Partir de Arquivo ---")

    target_letter = input(
        "Digite a letra do dataset que você quer usar para o teste (ex: A, B, J): "
    ).upper()
    csv_filename = os.path.join(DATA_DIR, f"dataset_{target_letter}.csv")

    if not os.path.exists(csv_filename):
        print(f"ERRO: O arquivo '{csv_filename}' não foi encontrado.")
        exit()

    df = pd.read_csv(csv_filename)

    available_samples = df["sample_id"].unique()
    print(f"\nAmostras disponíveis para a letra '{target_letter}':")
    print(list(available_samples))

    # ### INÍCIO DA CORREÇÃO ###
    # Converte o input do usuário para um número inteiro (int)
    try:
        target_sample_id = int(
            input(
                "Digite o ID da amostra que você quer testar (copie da lista acima): "
            )
        )
    except ValueError:
        print("ERRO: Você precisa digitar um número válido.")
        exit()
    # ### FIM DA CORREÇÃO ###

    if target_sample_id not in available_samples:
        print(f"ERRO: O sample_id '{target_sample_id}' não existe no arquivo.")
        exit()

    print(f"\nPreparando a amostra '{target_sample_id}'...")
    sample_data = df[df["sample_id"] == target_sample_id]

    sequence_to_predict = prepare_single_sequence(
        sample_data, FRAMES_PER_SAMPLE
    )
    normalized_sequence = normalize_single_sample(sequence_to_predict)

    print("Carregando modelo e realizando a predição...")
    model = keras.models.load_model(MODEL_PATH)
    classes = np.load(LABELS_PATH, allow_pickle=True)

    prediction_array = model.predict(normalized_sequence, verbose=0)[0]

    confidence = np.max(prediction_array)
    predicted_class_index = np.argmax(prediction_array)
    predicted_letter = classes[predicted_class_index]

    print("\n--- RESULTADO DA PREDIÇÃO ---")
    print(
        f"Amostra Testada: '{target_sample_id}' (Letra Real: {target_letter})"
    )
    print("-" * 30)
    print(f"Letra Prevista pelo Modelo: '{predicted_letter}'")
    print(f"Confiança da Predição: {confidence:.2%}")
    print("-" * 30)

    print("\nProbabilidades para cada classe:")
    for i, class_name in enumerate(classes):
        print(f"- {class_name}: {prediction_array[i]:.2%}")
        print(f"- {class_name}: {prediction_array[i]:.2%}")
