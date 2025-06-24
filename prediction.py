import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- CONFIGURAÇÕES ---
# Caminho para o modelo treinado e para o label encoder
MODEL_PATH = "modelo_todas_letras (7).keras"
LABELS_PATH = "label_encoder_classes.npy"
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.7

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- VALIDAÇÃO INICIAL ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(
        "ERRO CRÍTICO: Arquivos de modelo não encontrados. Rode 'trainingmodel.py' primeiro."
    )
    exit()

# --- INICIALIZAÇÃO ---
print("Carregando modelo e labels...")
model = keras.models.load_model(MODEL_PATH)
# Carrega as classes (letras) que o modelo pode prever
classes = np.load(LABELS_PATH)
print("Modelo e labels carregados com sucesso!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

print("Acessando a câmera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(2.0)

if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()
print(f"Câmera acessada! Resolução: {cap.get(3)}x{cap.get(4)}")

# --- VARIÁVEIS DE ESTADO ---
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions_deque = deque(maxlen=15)
is_writing_mode = False
written_text = ""
last_added_letter = ""
time_last_detection = time.time()

# --- CRIAÇÃO DA JANELA ---
WINDOW_NAME = "DatiloMaozinha - Predicao em Tempo Real"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

# --- LOOP PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_flipped = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_prediction_display = ""

    if result.multi_hand_landmarks:
        time_last_detection = time.time()
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Lógica de normalização e predição (continua a mesma)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        points = np.array(landmarks).reshape((21, 3))

        base_point = points[0].copy()
        points -= base_point
        max_value = np.max(np.linalg.norm(points, axis=1))
        if max_value > 1e-6:
            points /= max_value

        sequence.append(points)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)
            prediction_array = model.predict(input_data, verbose=0)[0]
            confidence = np.max(prediction_array)

            if confidence >= PREDICTION_THRESHOLD:
                predicted_class_index = np.argmax(prediction_array)
                predicted_letter = classes[predicted_class_index]
                predictions_deque.append(predicted_letter)

            if len(predictions_deque) > 5:
                most_common_prediction = max(
                    set(predictions_deque), key=predictions_deque.count
                )
                current_prediction_display = (
                    f"{most_common_prediction} ({confidence*100:.0f}%)"
                )

                if is_writing_mode:
                    if most_common_prediction != last_added_letter:
                        written_text += most_common_prediction
                        last_added_letter = most_common_prediction
                else:
                    written_text = most_common_prediction
    else:
        if time.time() - time_last_detection > 1.5:
            last_added_letter = ""
            predictions_deque.clear()

    # ### INÍCIO DA ALTERAÇÃO NA INTERFACE ###
    # --- EXIBIÇÃO NA TELA (COM AJUDA VISUAL) ---

    # Painel superior
    cv2.rectangle(frame_flipped, (0, 0), (FRAME_WIDTH, 60), (20, 20, 20), -1)
    cv2.putText(
        frame_flipped,
        f"PREVISAO: {current_prediction_display}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    mode_text = "MODO: ESCREVER" if is_writing_mode else "MODO: SOBRESCREVER"
    color = (0, 255, 0) if is_writing_mode else (0, 165, 255)
    (w, h), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame_flipped,
        mode_text,
        (FRAME_WIDTH - w - 10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    # Painel inferior
    cv2.rectangle(
        frame_flipped,
        (0, FRAME_HEIGHT - 100),
        (FRAME_WIDTH, FRAME_HEIGHT),
        (20, 20, 20),
        -1,
    )

    # Texto escrito
    cv2.putText(
        frame_flipped,
        written_text,
        (10, FRAME_HEIGHT - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 0),
        2,
    )

    # Novo: Texto de ajuda para os comandos
    help_text = "E: Escrever | S: Sobrescrever | D: Del | Espaco | Q: Sair"
    cv2.putText(
        frame_flipped,
        help_text,
        (10, FRAME_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    cv2.imshow(WINDOW_NAME, frame_flipped)
    # ### FIM DA ALTERAÇÃO NA INTERFACE ###

    # --- CONTROLE POR TECLADO ---
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("e"):
        if not is_writing_mode:
            is_writing_mode = True
            written_text = ""
            last_added_letter = ""
            print("MODO DE ESCRITA ATIVADO")
    elif key == ord("s"):
        if is_writing_mode:
            is_writing_mode = False
            print("MODO DE SOBRESCREVER ATIVADO")
    elif key == ord("d"):
        if is_writing_mode and len(written_text) > 0:
            written_text = written_text[:-1]
            last_added_letter = ""
    elif key == ord(" "):
        if is_writing_mode and (
            not written_text or not written_text.endswith(" ")
        ):
            written_text += " "
            last_added_letter = " "

# --- FINALIZAÇÃO ---
print("\nEncerrando o programa.")
cap.release()
cv2.destroyAllWindows()
