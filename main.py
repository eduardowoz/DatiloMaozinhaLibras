import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

# Configurações
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Dados de treinamento
gestos_salvos = {}  # Dicionário: {nome_gesto: [características]}
X_train = []  # Lista de features
y_train = []  # Lista de labels

# Inicializar classificador
knn = KNeighborsClassifier(n_neighbors=3)

def extrair_features(landmarks):
    """Extrai características normalizadas da mão"""
    features = []
    base_x, base_y = landmarks[0].x, landmarks[0].y
    
    for lm in landmarks:
        # Normalizar coordenadas relativas à base da mão
        features.extend([lm.x - base_x, lm.y - base_y])
    
    return features

# Variáveis de estado
tecla_t_pressionada = False
gesto_atual = ""
amostras_capturadas = 0
modo_reconhecimento = False

cap = cv2.VideoCapture(0)

nome_completo = ""
ultima_letra = ""
ultima_letra_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Verificar estado da tecla (usando polling mais preciso)
    key = cv2.waitKey(10) & 0xFF
    
    # Lógica de captura com tecla mantida
    if key == ord('t'):
        if not tecla_t_pressionada:
            gesto_atual = input("Digite o nome do gesto (e pressione Enter): ")
            tecla_t_pressionada = True
            amostras_capturadas = 0
            print(f"Capturando gesto '{gesto_atual}'... (Mantenha 'T' pressionada)")
        
        # Captura contínua enquanto 'T' está pressionada
        if tecla_t_pressionada and results.multi_hand_landmarks:
            features = extrair_features(results.multi_hand_landmarks[0].landmark)
            
            if gesto_atual not in gestos_salvos:
                gestos_salvos[gesto_atual] = []
            
            gestos_salvos[gesto_atual].append(features)
            X_train.append(features)
            y_train.append(gesto_atual)
            amostras_capturadas += 1
            
            # Feedback visual
            cv2.putText(image, f"Capturando: '{gesto_atual}'", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Amostras: {amostras_capturadas}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        if tecla_t_pressionada:  # Tecla 'T' foi liberada
            tecla_t_pressionada = False
            if len(set(y_train)) >= 2:
                knn.fit(X_train, y_train)
                print(f"Treinamento concluído! Amostras de '{gesto_atual}': {amostras_capturadas}")
            else:
                print("Adicione pelo menos 2 gestos diferentes para treinar o modelo!")
    
    # Modo de reconhecimento
    if key == ord('r'):
        modo_reconhecimento = True
        print("Modo reconhecimento ativo!")
    
    if modo_reconhecimento and results.multi_hand_landmarks and len(X_train) > 0:
        features = extrair_features(results.multi_hand_landmarks[0].landmark)
        letra_atual = knn.predict([features])[0]
        
        # Lógica para formar o nome
        if letra_atual != ultima_letra:
            ultima_letra = letra_atual
            ultima_letra_time = time.time()
        elif time.time() - ultima_letra_time > 2:  # 2 segundos na mesma letra
            nome_completo += letra_atual
            ultima_letra = ""
        
        # Exibir informações
        cv2.putText(image, f"Letra: {letra_atual}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Nome: {nome_completo}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Desenhar landmarks
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    
    # Menu de ajuda
    cv2.putText(image, "T: Treinar (mantenha pressionado) | R: Reconhecer | Q: Sair", 
               (10, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Aprendizado de Gestos', image)
    
    if key == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()