import os

import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


# 1. Carregar e concatenar todos os datasets
def load_and_combine_datasets(base_dir):
    all_dfs = []
    classes = []

    for filename in os.listdir(base_dir):
        if filename.endswith(".csv"):
            letter = filename.split("_")[-1].split(".")[0]
            filepath = os.path.join(base_dir, filename)

            df = pd.read_csv(filepath)

            # Garante que sample_id seja único entre datasets
            df["sample_id"] = letter + "_" + df["sample_id"].astype(str)

            df["label"] = letter
            all_dfs.append(df)
            classes.append(letter)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, sorted(list(set(classes)))


# Modificação no pré-processamento para agrupar frames por sample_id
def prepare_sequences(data, n_frames=30):
    samples = []
    labels = []

    # Agrupa frames por sample_id
    grouped = data.groupby("sample_id")

    for name, group in grouped:
        # Pega os primeiros n_frames (ou preenche com zeros se menor)
        frames = group.drop(["sample_id", "frame_id", "label"], axis=1).values
        if len(frames) < n_frames:
            # Padding com zeros se necessário
            pad = np.zeros((n_frames - len(frames), frames.shape[1]))
            frames = np.vstack([frames, pad])
        else:
            frames = frames[:n_frames]

        # Reshape para (n_frames, 21, 3)
        frames = frames.reshape(n_frames, 21, 3)
        samples.append(frames)
        labels.append(group["label"].iloc[0])  # Pega o label do sample

    return np.array(samples), np.array(labels)


# Configurações
DATA_DIR = "dataset_libras"
combined_data, classes = load_and_combine_datasets(DATA_DIR)
MODEL_PATH = "modelo_todas_letras.keras"

# 2. Pré-processamento corrigido
X, y = prepare_sequences(combined_data, n_frames=30)  # 30 frames por amostra

# Normalização por coordenada
# for i in range(X.shape[0]):  # Para cada amostra
#     for j in range(21):  # Para cada ponto
#         for k in range(3):  # Para cada coordenada (x,y,z)
#             min_val = X[:, :, j, k].min()
#             max_val = X[:, :, j, k].max()
#             X[i, :, j, k] = (X[i, :, j, k] - min_val) / (
#                 max_val - min_val + 1e-8
#             )

# Codificação das classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = keras.utils.to_categorical(y_encoded, num_classes=len(classes))


# 3. Arquitetura do Modelo com Hiperparametrização similar
def build_model(hp):
    model = keras.Sequential()

    # Input: (n_frames, 21, 3)
    model.add(keras.layers.Input(shape=(30, 21, 3)))

    # Bloco Convolucional TimeDistributed
    for i in range(hp.Int("num_conv_layers", 1, 2)):
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Conv1D(
                    filters=hp.Int(f"conv_filters_{i}", 64, 256, step=64),
                    kernel_size=hp.Choice(f"conv_kernel_{i}", [3, 5]),
                    activation=hp.Choice(
                        f"conv_activation_{i}", ["relu", "swish"]
                    ),
                    padding="same",
                )
            )
        )
        model.add(
            keras.layers.TimeDistributed(keras.layers.BatchNormalization())
        )
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.MaxPooling1D(
                    pool_size=hp.Int(f"pool_size_{i}", 2, 3)
                )
            )
        )

    # Achatar os features para cada timestep
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

    # Bloco LSTM
    lstm_units = hp.Int("lstm_units", 64, 256, step=64)
    return_sequences = hp.Boolean("lstm_return_seq")

    if hp.Boolean("use_bilstm"):
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=lstm_units, return_sequences=return_sequences
                )
            )
        )
    else:
        model.add(
            keras.layers.LSTM(
                units=lstm_units, return_sequences=return_sequences
            )
        )

    if return_sequences:
        model.add(keras.layers.GlobalAveragePooling1D())

    model.add(
        keras.layers.Dropout(rate=hp.Float("dropout_rate", 0.1, 0.5, step=0.1))
    )

    # Camadas Densas
    for i in range(hp.Int("num_dense_layers", 1, 2)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"dense_units_{i}", 32, 256, step=32),
                activation=hp.Choice(
                    f"dense_activation_{i}", ["relu", "gelu"]
                ),
            )
        )
        if hp.Boolean(f"use_dense_dropout_{i}"):
            model.add(
                keras.layers.Dropout(
                    rate=hp.Float(f"dense_dropout_{i}", 0.1, 0.3, step=0.1)
                )
            )

    model.add(keras.layers.Dense(len(classes), activation="softmax"))

    optimizer = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model.compile(
        optimizer=keras.optimizers.get(
            {
                "class_name": optimizer,
                "config": {"learning_rate": learning_rate},
            }
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# 4. Configuração do K-Fold para busca de hiperparâmetros
kfold_search = KFold(n_splits=3, shuffle=True, random_state=42)
batch_size = 32

# 5. Busca de Hiperparâmetros com Cross-Validation
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=30,
    directory="tuner_results",
    project_name="sign_language_tuning",
    overwrite=True,
)

for fold, (train_idx, val_idx) in enumerate(kfold_search.split(X, y_onehot)):
    print(f"\nFold {fold + 1} - Busca de Hiperparâmetros")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

    tuner.search(
        X_train,
        y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=5,
                monitor="val_accuracy",
                mode="max",
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

# 6. Avaliação Final com Cross-Validation
if len(tuner.get_best_hyperparameters()) > 0:
    best_hps = tuner.get_best_hyperparameters()[0]
    print("\nMelhores hiperparâmetros encontrados:")
    for param, value in best_hps.values.items():
        print(f"- {param}: {value}")

    # Configurar K-Fold para avaliação final (usando mais folds)
    kfold_eval = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []
    histories = []

    for train_idx, test_idx in kfold_eval.split(X, y_onehot):
        print(f"\nFold {fold_no} - Avaliação Final")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

        # Criar e treinar modelo com melhores hiperparâmetros
        model = tuner.hypermodel.build(best_hps)

        history = model.fit(
            X_train,
            y_train,
            epochs=2000,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    monitor="accuracy",
                    mode="max",
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="accuracy", factor=0.5, patience=3, min_lr=1e-6
                ),
            ],
            verbose=2,
        )

        # Avaliar no fold de teste
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)
        histories.append(history)

        print(f"Acurácia no Fold {fold_no}: {accuracy:.4f}")
        fold_no += 1

    # Resultados da avaliação cruzada
    print("\nResultados da Avaliação Cruzada:")
    print(f"Acurácias por fold: {[round(acc, 4) for acc in accuracies]}")
    print(
        f"Acurácia média: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})"
    )

    # Treinar modelo final com todos os dados (opcional)
    print("\nTreinando modelo final com todos os dados...")
    final_model = tuner.hypermodel.build(best_hps)
    final_model.fit(
        X,
        y_onehot,
        epochs=200,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10,
                monitor="accuracy",
                mode="max",
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
            ),
        ],
        verbose=2,
    )

    # Salvar modelo e label encoder
    final_model.save(MODEL_PATH)
    np.save("label_encoder_classes.npy", le.classes_)
    print("Modelo e label encoder salvos com sucesso!")
else:
    print("Nenhum modelo válido foi encontrado durante a otimização.")


# ###########################################################

# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow import keras
# import keras_tuner as kt

# # --- FUNÇÕES DE PREPARAÇÃO ---

# def load_and_combine_datasets(base_dir):
#     """Carrega todos os arquivos CSV de um diretório e os concatena."""
#     all_dfs = []
#     classes = []

#     if not os.path.exists(base_dir):
#         print(f"ERRO: O diretório de datasets '{base_dir}' não foi encontrado.")
#         return pd.DataFrame(), []

#     for filename in os.listdir(base_dir):
#         if filename.endswith(".csv"):
#             letter = filename.split("_")[-1].split(".")[0]
#             filepath = os.path.join(base_dir, filename)
#             try:
#                 df = pd.read_csv(filepath)
#                 if not df.empty:
#                     # A label já está no CSV da captura, mas garantimos aqui
#                     if "label" not in df.columns:
#                         df["label"] = letter
#                     all_dfs.append(df)
#                     if letter not in classes:
#                         classes.append(letter)
#             except pd.errors.EmptyDataError:
#                 print(f"AVISO: O arquivo '{filename}' está vazio e será ignorado.")
#                 continue

#     if not all_dfs:
#         return pd.DataFrame(), []

#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     return combined_df, sorted(classes)

# def prepare_sequences(data, n_frames):
#     """Agrupa os dados por amostra e formata em sequências de n_frames."""
#     samples, labels = [], []
#     grouped = data.groupby("sample_id")

#     for _, group in grouped:
#         frames = group.drop(["sample_id", "frame_id", "label"], axis=1).values
#         if len(frames) < n_frames:
#             pad = np.zeros((n_frames - len(frames), frames.shape[1]))
#             frames = np.vstack([frames, pad])
#         else:
#             frames = frames[:n_frames]
#         samples.append(frames)
#         labels.append(group["label"].iloc[0])

#     return np.array(samples), np.array(labels)

# def normalize_dataset(X):
#     """Aplica a normalização (pulso + escala) em todo o dataset."""
#     X_reshaped = X.reshape(X.shape[0], X.shape[1], 21, 3)
#     X_normalized = np.zeros_like(X_reshaped, dtype=float)
#     for i in range(X_reshaped.shape[0]):
#         for j in range(X_reshaped.shape[1]):
#             frame_landmarks = X_reshaped[i, j]
#             base_point = frame_landmarks[0].copy()
#             normalized_frame = frame_landmarks - base_point
#             max_value = np.max(np.linalg.norm(normalized_frame, axis=1))
#             if max_value > 1e-6:
#                 normalized_frame /= max_value
#             X_normalized[i, j] = normalized_frame
#     return X_normalized

# # --- CONFIGURAÇÕES ---
# DATA_DIR = "dataset_libras"
# MODEL_PATH = "modelo_todas_letras.keras"
# LABELS_PATH = "label_encoder_classes.npy"
# FRAMES_PER_SAMPLE = 30

# # --- ETAPA 1: CARREGAMENTO DOS DADOS ---
# print("--- Etapa 1: Carregando e preparando os dados ---")
# combined_data, classes = load_and_combine_datasets(DATA_DIR)

# if combined_data.empty:
#     print("Nenhum dado para treinar. Verifique a pasta de datasets. Encerrando.")
#     exit()

# # Adicionado para printar as classes encontradas
# print(f"Classes encontradas para treinamento: {classes}")

# X, y = prepare_sequences(combined_data, n_frames=FRAMES_PER_SAMPLE)
# print(f"Total de amostras preparadas: {len(X)}")

# # --- ETAPA 2: NORMALIZAÇÃO E CODIFICAÇÃO ---
# print("\n--- Etapa 2: Normalizando dados e codificando labels ---")

# # Reativando a normalização correta e consistente
# X = normalize_dataset(X)
# print("Normalização concluída!")

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_onehot = keras.utils.to_categorical(y_encoded, num_classes=len(classes))
# np.save(LABELS_PATH, le.classes_)
# print(f"Labels salvos em '{LABELS_PATH}'")

# # --- ETAPA 3: CONSTRUÇÃO DO MODELO (KERAS TUNER) ---
# def build_model(hp):
#     model = keras.Sequential()
#     model.add(keras.layers.Input(shape=(FRAMES_PER_SAMPLE, 21, 3)))
#     for i in range(hp.Int("num_conv_layers", 1, 2, default=1)):
#         model.add(keras.layers.TimeDistributed(
#             keras.layers.Conv1D(
#                 filters=hp.Int(f"conv_filters_{i}", 32, 128, step=32),
#                 kernel_size=hp.Choice(f"conv_kernel_{i}", [3, 5]),
#                 activation="relu", padding="same")))
#         model.add(keras.layers.TimeDistributed(keras.layers.BatchNormalization()))
#         model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2)))
#     model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
#     lstm_units = hp.Int("lstm_units", 64, 256, step=64)
#     model.add(keras.layers.LSTM(units=lstm_units, return_sequences=False))
#     model.add(keras.layers.Dropout(rate=hp.Float("dropout_rate", 0.2, 0.5, step=0.1)))
#     for i in range(hp.Int("num_dense_layers", 1, 2, default=1)):
#         model.add(keras.layers.Dense(
#             units=hp.Int(f"dense_units_{i}", 64, 256, step=64),
#             activation="relu"))
#     model.add(keras.layers.Dense(len(classes), activation="softmax"))
#     learning_rate = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log")
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"])
#     return model

# # --- ETAPA 4: BUSCA DE HIPERPARÂMETROS ---
# print("\n--- Etapa 4: Buscando os melhores hiperparâmetros ---")
# X_train_full, X_tuner_val, y_train_full, y_tuner_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
# tuner = kt.Hyperband(
#     build_model,
#     objective="val_accuracy",
#     max_epochs=30,
#     factor=3,
#     directory="tuner_results",
#     project_name="sign_language_tuning",
#     overwrite=True)
# tuner.search(X_train_full, y_train_full, epochs=50, validation_data=(X_tuner_val, y_tuner_val),
#              callbacks=[keras.callbacks.EarlyStopping(patience=10, monitor="val_loss")])
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print("\nMelhores hiperparâmetros encontrados:")
# print(best_hps.values)

# # --- ETAPA 5: AVALIAÇÃO FINAL COM K-FOLD ---
# print("\n--- Etapa 5: Avaliando o melhor modelo com K-Fold ---")
# kfold_eval = KFold(n_splits=5, shuffle=True, random_state=42)
# accuracies = []
# fold_no = 1
# for train_idx, test_idx in kfold_eval.split(X, y_onehot):
#     print("-" * 50)
#     print(f"Treinando Fold {fold_no}...")
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]
#     model = tuner.hypermodel.build(best_hps)
#     history = model.fit(
#         X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
#         callbacks=[
#             keras.callbacks.EarlyStopping(patience=15, monitor="val_accuracy", mode="max", restore_best_weights=True),
#             keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=5)
#         ], verbose=0)
#     loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#     accuracies.append(accuracy)
#     print(f"Acurácia no Fold {fold_no}: {accuracy:.4f}")
#     fold_no += 1
# print("-" * 50)
# print(f"Acurácia Média Final (K-Fold): {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
# print("-" * 50)

# # --- ETAPA 6: TREINAMENTO FINAL COM TODOS OS DADOS ---
# print("\n--- Etapa 6: Treinando o modelo final com todos os dados ---")
# final_model = tuner.hypermodel.build(best_hps)
# final_model.fit(
#     X, y_onehot, epochs=200, batch_size=32, validation_split=0.1,
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=20, monitor="val_accuracy", mode="max", restore_best_weights=True),
#         keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max")
#     ], verbose=1)
# print(f"\nModelo final salvo em '{MODEL_PATH}'")
# print("Processo de treinamento concluído!")
