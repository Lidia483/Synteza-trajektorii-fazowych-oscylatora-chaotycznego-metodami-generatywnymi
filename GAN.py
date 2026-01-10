import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Reshape
from tensorflow.keras.optimizers import Adam
import warnings



# --- KONFIGURACJA ---
SEQ_LEN = 50           # Długość historii 
PREDICT_STEPS = 2000   # Długość generowanego sygnału 
TRAIN_SIZE = 20000     # Ile danych bierzemy z pliku
NOISE_DIM = 100        # Wielkość szumu
EPOCHS = 1000          # Liczba rund treningu
BATCH_SIZE = 32
TIMESTEP = 2.5e-6

# Wczytanie danych
print("Wczytywanie danych...")
try:
    df = pd.read_csv('foo.csv', header=None)
    limit = min(TRAIN_SIZE, len(df))
    time_vals = df.iloc[:limit, 0].values
    signal = df.iloc[:limit, 1:3].values
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

# --- PRZYGOTOWANIE DANYCH DO TRENINGU GAN ---
gan_seq_len = PREDICT_STEPS 
signal_norm = signal 
required_len = gan_seq_len + (BATCH_SIZE * 50)
temp_signal = signal_norm

# Powielanie danych jeśli plik jest za krótki
while len(temp_signal) < required_len:
    print(f"Dane są za krótkie ({len(temp_signal)}). Powielam je...")
    temp_signal = np.concatenate([temp_signal, signal_norm])

signal_train = temp_signal

# Tworzenie zbioru treningowego dla GAN
X_gan_train = []
step = 50 
for i in range(0, len(signal_train) - gan_seq_len, step):
    X_gan_train.append(signal_train[i : i + gan_seq_len])

X_gan_train = np.array(X_gan_train)

# Dopasowanie do Batch Size
if len(X_gan_train) < BATCH_SIZE:
    repeat_count = (BATCH_SIZE // len(X_gan_train)) + 1
    X_gan_train = np.concatenate([X_gan_train] * repeat_count)

train_len = (len(X_gan_train) // BATCH_SIZE) * BATCH_SIZE
X_gan_train = X_gan_train[:train_len]

print(f"Dane treningowe gotowe. Liczba sekwencji: {len(X_gan_train)}")

# --- BUDOWA SIECI ---

def build_generator():
    inp = Input(shape=(NOISE_DIM,))
    x = Dense(PREDICT_STEPS * 2)(inp) # Używamy PREDICT_STEPS
    x = Reshape((PREDICT_STEPS, 2))(x)
    x = Dense(2, activation='tanh')(x)
    return Model(inp, x)

def build_discriminator():
    inp = Input(shape=(PREDICT_STEPS, 2))
    x = LSTM(50, return_sequences=False)(inp)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

discriminator = build_discriminator()
generator = build_generator()

discriminator.trainable = False  
generator.trainable = True       

gan_input = Input(shape=(NOISE_DIM,))
fake_signal = generator(gan_input)
gan_output = discriminator(fake_signal)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Trening
print("\nStart Treningu...")
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    idx = np.random.randint(0, X_gan_train.shape[0], BATCH_SIZE)
    real_seqs = X_gan_train[idx]
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    gen_seqs = generator.predict(noise, verbose=0)
    
    discriminator.train_on_batch(real_seqs, real_labels)
    discriminator.train_on_batch(gen_seqs, fake_labels)
    
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = gan.train_on_batch(noise, real_labels)
    
    if epoch % 100 == 0:
        loss_val = g_loss[0] if isinstance(g_loss, list) else g_loss
        print(f"Runda {epoch}/{EPOCHS} | Strata Generatora: {loss_val:.4f}")

# Prognozowanie
print("\nGenerowanie przebiegu...")
noise = np.random.normal(0, 1, (1, NOISE_DIM))
generated_norm = generator.predict(noise, verbose=0)[0]
predictions = generated_norm

X_train = [signal[:SEQ_LEN]] 
X_train = np.array(X_train) 

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
