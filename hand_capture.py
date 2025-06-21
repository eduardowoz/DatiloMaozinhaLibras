# hand_capture.py (versão com tratamento de EmptyDataError)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# --- Configurações ---
DATA_DIR = 'dataset_libras'
FRAMES_PER_SAMPLE = 30

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# --- Preparação do ambiente ---
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Pasta do dataset '{DATA_DIR}' pronta.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# --- Interação com o Usuário ---
label = input("Digite a letra que sera capturada: ").upper()
samples_to_capture = int(input(f"Quantas amostras da letra '{label}' voce quer capturar? "))
output_filename = os.path.join(DATA_DIR, f'dataset_{label}.csv')

# --- Lógica de Carregamento/Criação de arquivo (MAIS ROBUSTA) ---
all_data = []
start_sample_id = 0

try:
    # Tenta ler o CSV. Se não existir ou estiver vazio, vai pular para o 'except'.
    existing_df = pd.read_csv(output_filename)
    if not existing_df.empty:
        all_data = existing_df.values.tolist()
        start_sample_id = existing_df['sample_id'].max() + 1
        print(f"Arquivo '{output_filename}' encontrado. Novas amostras serao adicionadas.")
    else:
        # O arquivo existe mas só tem cabeçalho, por exemplo.
        print(f"Arquivo '{output_filename}' encontrado, mas esta vazio. Iniciando nova coleta.")
except (FileNotFoundError, pd.errors.EmptyDataError):
    # Trata os dois casos: arquivo não existe OU existe mas está 100% vazio.
    pass # Não faz nada, pois as variáveis 'all_data' e 'start_sample_id' já estão prontas para um novo arquivo.


print(f"\nOK! Coletando para a letra '{label}'. O arquivo de saida sera '{output_filename}'.")
time.sleep(1)

# --- Loop Principal de Captura ---
collected_samples = 0
quit_flag = False

while collected_samples < samples_to_capture:
    # Espera o usuário pressionar a tecla para iniciar a captura da amostra
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        
        if all_data:
            # Calcula o total de amostras únicas já existentes
            temp_df = pd.DataFrame(all_data)
            # Verifica se a coluna 'sample_id' (índice 0) existe antes de contar
            if 0 in temp_df.columns:
                total_samples_in_file = len(temp_df[0].unique())
            else:
                total_samples_in_file = 0
        else:
            total_samples_in_file = 0
            
        cv2.putText(frame, f"LETRA: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Amostras a coletar nesta sessao: {collected_samples}/{samples_to_capture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Amostras totais no arquivo: {total_samples_in_file}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Segure 'ESPACO' para capturar", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Pressione 'Q' para finalizar e salvar", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Captura de Gestos', frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):
            current_sample_id = start_sample_id + collected_samples
            print(f"\nIniciando captura da amostra {current_sample_id} para a letra '{label}'...")
            break
        elif key == ord('q'):
            quit_flag = True
            break
    
    if quit_flag:
        break

    # Inicia a captura da sequência de frames
    for frame_num in range(FRAMES_PER_SAMPLE):
        ret, frame = cap.read()
        if not ret:
            print("Erro no frame, pulando...")
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        cv2.putText(frame, f"GRAVANDO... Frame {frame_num + 1}/{FRAMES_PER_SAMPLE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        landmarks_data = []
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            points = np.array(landmarks).reshape((21, 3))
            base_point = points[0].copy()
            points -= base_point
            
            max_value = np.max(np.linalg.norm(points, axis=1))
            if max_value > 1e-6:
                points /= max_value
            
            landmarks_data = points.flatten().tolist()
        else:
            landmarks_data = [0.0] * 63

        row = [current_sample_id, frame_num, label] + landmarks_data
        all_data.append(row)

        cv2.imshow('Captura de Gestos', frame)
        cv2.waitKey(1)
            
    collected_samples += 1
    print(f"Amostra {current_sample_id} para '{label}' capturada com sucesso!")
    time.sleep(0.5)

print("\n>>> Finalizando captura...")

# --- Salvamento dos dados ---
if all_data:
    columns = ['sample_id', 'frame_id', 'label']
    for i in range(21):
        columns += [f'x{i}', f'y{i}', f'z{i}']
        
    df = pd.DataFrame(all_data, columns=columns)
    df.drop_duplicates(inplace=True) 
    df.to_csv(output_filename, index=False)
    
    print(f"\nDataset salvo com sucesso em '{output_filename}'")
    print(f"Total de amostras no arquivo: {df['sample_id'].nunique()}")
    print(f"Total de frames (linhas no CSV): {len(df)}")
else:
    print("Nenhuma nova amostra foi capturada.")

# --- Encerrando ---
cap.release()
cv2.destroyAllWindows()