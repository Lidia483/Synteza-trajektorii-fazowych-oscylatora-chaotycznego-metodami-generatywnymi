import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Input, Dropout, GaussianNoise
from sklearn.preprocessing import MinMaxScaler

# 1. KONFIGURACJA
folder_name = "Wyniki_LSTM"
if not os.path.exists(folder_name): os.makedirs(folder_name)

# Parametry (Analogiczne do GAN)
SEQ_LEN = 64            
PREDICT_STEPS = 2000    
EPOCHS = 40            
BATCH_SIZE = 32         
TIMESTEP = 2.5e-6       

# 2. DANE
print("Wczytywanie danych...")
try:
    df = pd.read_csv('foo.csv', header=None)
    data_len = len(df)
    
    if data_len < 5000:
        DOWNSAMPLE = 1
        print("Mały zbiór danych - pobieranie wszystkich próbek.")
    else:
        DOWNSAMPLE = 5
        print(f"Duży zbiór danych - pobieranie co {DOWNSAMPLE} próbki.")

    # Pobranie danych (kolumny 1 i 2) - identycznie jak w GAN
    signal_raw = df.iloc[::DOWNSAMPLE, 1:3].values
    
except FileNotFoundError:
    print("Błąd: Brak pliku foo.csv")
    exit()

# Skalowanie (Zakres -1 do 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
signal_scaled = scaler.fit_transform(signal_raw)

X, y = [], []
for i in range(len(signal_scaled) - SEQ_LEN):
    X.append(signal_scaled[i : i + SEQ_LEN])
    y.append(signal_scaled[i + SEQ_LEN])

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)

# 3. BUDOWA MODELU LSTM
model = Sequential([
    Input(shape=(SEQ_LEN, 2)),
    GaussianNoise(0.05), 
    LSTM(128, return_sequences=True),
    LeakyReLU(negative_slope=0.2),
    Dropout(0.2),
    LSTM(64),
    LeakyReLU(negative_slope=0.2),
    Dense(2, activation='tanh') 
])

model.compile(optimizer='adam', loss='mse')

print("Start treningu...")
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# 4. GENEROWANIE
print("Generowanie przebiegu...")
curr_seq = X[-1].reshape(1, SEQ_LEN, 2) 
history_seq = curr_seq.copy()           

preds = []
for i in range(PREDICT_STEPS):
    next_point = model.predict(curr_seq, verbose=0)
    preds.append(next_point[0])
    # Przesunięcie okna o jeden krok
    curr_seq = np.concatenate([curr_seq[:, 1:, :], next_point.reshape(1, 1, 2)], axis=1)
    if i % 100 == 0: print(f"Krok {i}/{PREDICT_STEPS}")

preds = np.array(preds)
preds_inv = scaler.inverse_transform(preds)
hist_inv = scaler.inverse_transform(history_seq[0])

ts = datetime.datetime.now().strftime("%H%M%S")

# 5. WIZUALIZACJA (Analogicznie do GAN)

# WYKRES 1: ATRAKTOR
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(signal_raw[:PREDICT_STEPS, 0], signal_raw[:PREDICT_STEPS, 1], 'b', alpha=0.3, lw=0.5)
plt.title("Oryginał")
plt.xlabel("V1 [-1, 1]")
plt.ylabel("V2 [-1, 1]")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(preds_inv[:, 0], preds_inv[:, 1], 'r', alpha=0.6, lw=0.5)
plt.title("Prognoza LSTM")
plt.xlabel("V1 [-1, 1]")
plt.ylabel("V2 [-1, 1]")
plt.grid(True)

plt.savefig(os.path.join(folder_name, f"Atraktor_{ts}.png"))
plt.pause(3)
plt.close()

# WYKRES 2: PRZEBIEG CZASOWY 
plt.figure(figsize=(15, 6))
t_hist = np.arange(SEQ_LEN) * TIMESTEP
t_pred = np.arange(SEQ_LEN, SEQ_LEN + PREDICT_STEPS) * TIMESTEP

plt.plot(t_hist, hist_inv[:, 0], 'b', label='Historia V1')
plt.plot(t_hist, hist_inv[:, 1], 'g', label='Historia V2')
plt.plot(t_pred, preds_inv[:, 0], 'r', alpha=0.8, label='Prognoza V1')
plt.plot(t_pred, preds_inv[:, 1], 'm', alpha=0.8, label='Prognoza V2')

plt.axvline(x=SEQ_LEN*TIMESTEP, color='k', ls='--', label='Start Generacji')

plt.title("Przebieg czasowy sygnałów (LSTM)")
plt.xlabel("Czas [ms]")
plt.ylabel("Amplituda [-1, 1]")

plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(folder_name, f"Sygnal_{ts}.png"))
plt.pause(3)
plt.close()

print(f"\nGotowe! Wyniki zapisane w folderze: {folder_name}")
