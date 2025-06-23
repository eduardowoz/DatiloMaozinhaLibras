import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from collections import deque

# --- CONFIGURAÇÕES ---
# Caminho para o modelo treinado e para o label encoder
MODEL_PATH = "modelo_todas_letras.keras"
LABELS_PATH = "label_encoder_classes.npy"

# Comprimento da sequência de frames que o modelo espera
SEQUENCE_LENGTH = 30

# Limiar de confiança para exibir a predição
PREDICTION_THRESHOLD = 0.8  # 80% de confiança

# --- INICIALIZAÇÃO ---
print("Carregando modelo e labels...")
# Carrega o modelo de deep learning treinado
model = keras.models.load_model(MODEL_PATH)
# Carrega as classes (letras) que o modelo pode prever
classes = np.load(LABELS_PATH)
print("Modelo e labels carregados com sucesso!")

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Fila para armazenar os frames (sequência de landmarks)
sequence = deque(maxlen=SEQUENCE_LENGTH)
# Fila para suavizar as predições
predictions_deque = deque(maxlen=10) # Armazena as últimas 10 predições

current_prediction = ""

# --- LOOP PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame. Encerrando...")
        break

    # Espelha a imagem para uma visualização tipo "espelho"
    frame = cv2.flip(frame, 1)
    # Converte a imagem de BGR para RGB, que é o formato esperado pelo MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem para detectar as mãos
    result = hands.process(rgb_frame)

    landmarks_data = []
    # Se uma mão for detectada
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        # Desenha os landmarks e as conexões na imagem
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- NORMALIZAÇÃO DOS LANDMARKS (igual ao script de captura) ---
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        points = np.array(landmarks).reshape((21, 3))
        
        # 1. Relativizar ao ponto do pulso (landmark 0)
        base_point = points[0].copy()
        points -= base_point
        
        # 2. Normalizar pela distância máxima
        max_value = np.max(np.linalg.norm(points, axis=1))
        if max_value > 1e-6: # Evita divisão por zero
            points /= max_value
        
        # Adiciona os pontos normalizados à sequência
        sequence.append(points)

        # --- PREDIÇÃO ---
        # Verifica se a sequência atingiu o tamanho necessário para o modelo
        if len(sequence) == SEQUENCE_LENGTH:
            # Converte a sequência para o formato que o modelo espera: (1, 30, 21, 3)
            input_data = np.expand_dims(np.array(sequence), axis=0)
            input_data = input_data.reshape(1, SEQUENCE_LENGTH, 21, 3) # Garante o reshape correto

            # Realiza a predição
            prediction = model.predict(input_data, verbose=0)[0]
            
            confidence = np.max(prediction)
            
            # Adiciona a predição na fila de suavização
            if confidence >= PREDICTION_THRESHOLD:
                predicted_class_index = np.argmax(prediction)
                predicted_letter = classes[predicted_class_index]
                predictions_deque.append(predicted_letter)
            
            # Atualiza a predição exibida se ela for a mais comum nas últimas frames
            if len(predictions_deque) > 0:
                most_common_prediction = max(set(predictions_deque), key=predictions_deque.count)
                current_prediction = f"{most_common_prediction} ({confidence*100:.1f}%)"
    else:
        # Se nenhuma mão for detectada, limpa a sequência para evitar predições antigas
        # sequence.clear() # Opcional: pode tornar a detecção mais "imediata" ao reaparecer a mão
        pass

    # --- EXIBIÇÃO NA TELA ---
    # Coloca um retângulo de fundo para o texto
    cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
    # Escreve a predição na tela
    cv2.putText(
        frame,
        f"LETRA: {current_prediction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(frame, "Pressione 'Q' para sair", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Predicao de LIBRAS em Tempo Real', frame)

    # Verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --- FINALIZAÇÃO ---
print("\nEncerrando o programa.")
cap.release()
cv2.destroyAllWindows()