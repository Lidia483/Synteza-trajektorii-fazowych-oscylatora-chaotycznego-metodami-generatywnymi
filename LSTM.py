import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- KONFIGURACJA ---
SEQ_LEN = 50           # Długość historii 
TRAIN_SIZE = 20000     # Ile danych bierzemy
PREDICT_STEPS = 2000   # Ile danych bierzemy z pliku
TIMESTEP = 2.5e-6 

# Wczytanie danych
print("Wczytywanie danych...")
try:
    df = pd.read_csv('foo.csv', header=None)
    time_vals = df.iloc[:TRAIN_SIZE, 0].values
    signal = df.iloc[:TRAIN_SIZE, 1:3].values
except FileNotFoundError:
    print("BŁĄD: Brak pliku foo.csv")
    exit()    


# Wizualizacja wejścia
plt.figure(figsize=(12, 5))
plt.plot(time_vals[:1000], signal[:1000, 0], 'b', label='V(v1) [ChL]')
plt.plot(time_vals[:1000], signal[:1000, 1], 'g', label='V(v2) [ChR]')
plt.title('Podgląd danych wejściowych')
plt.legend()
plt.grid(True)
print("Wyświetlam dane wejściowe. Okno pozostanie otwarte w tle.")
plt.show(block=False)  
plt.pause(3)        

# Tworzenie sekwencji
X, y = [], []
for i in range(len(signal) - SEQ_LEN):
    X.append(signal[i : i + SEQ_LEN])
    y.append(signal[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, 2)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(2)
])
model.compile(optimizer='adam', loss='mse')

# Trening
print("\nStart Treningu...")
history = model.fit(
    X_train, y_train,
    epochs=10, 
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Prognozowanie
print("\nGenerowanie prognozy...")
curr_seq = X_train[-1].reshape(1, SEQ_LEN, 2)
predictions = []

for _ in range(PREDICT_STEPS):
    pred = model.predict(curr_seq, verbose=0)[0]
    predictions.append(pred)
    new_step = pred.reshape(1, 1, 2)
    curr_seq = np.append(curr_seq[:, 1:, :], new_step, axis=1)

predictions = np.array(predictions)

# Wykres
plt.figure(figsize=(15, 6))
t_hist = np.arange(SEQ_LEN) * TIMESTEP
t_pred = np.arange(SEQ_LEN, SEQ_LEN + PREDICT_STEPS) * TIMESTEP

plt.plot(t_hist, X_train[-1][:, 0], 'b', label='Otrzymany przebieg V(v1) [ChL]')
plt.plot(t_hist, X_train[-1][:, 1], 'g', label='Otrzymany przebieg V(v2) [ChR]')
plt.plot(t_pred, predictions[:, 0], 'r--', label='Prognoza V(v1) [ChL]')
plt.plot(t_pred, predictions[:, 1], 'm--', label='Prognoza V(v2) [ChR]')

plt.axvline(x=SEQ_LEN * TIMESTEP, color='k', linestyle=':', label='Start prognozy')

# --- PUNKT 0.003s ---
target_time = 0.003
idx = (np.abs(t_pred - target_time)).argmin()
val_L = predictions[idx, 0]
val_R = predictions[idx, 1]
time_point = t_pred[idx]

print(f"\nPunkt kontrolny dla czasu {target_time}s:")
print(f"Czas: {time_point:.6f} s")
print(f"Współrzędna Y (Lewy): {val_L:.6f}")
print(f"Współrzędna Y (Prawy): {val_R:.6f}")

plt.plot(time_point, val_L, 'ro', markersize=8)
plt.text(time_point, val_L, f'  Y={val_L:.5f}', color='black', fontweight='bold', verticalalignment='bottom')

plt.plot(time_point, val_R, 'mo', markersize=8)
plt.text(time_point, val_R, f'  Y={val_R:.5f}', color='black', fontweight='bold', verticalalignment='top')

plt.title(f'Prognoza przebiegu chaotycznego')
plt.legend()
plt.grid(True)
plt.show()
