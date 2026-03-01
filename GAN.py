import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, LeakyReLU, Concatenate, GaussianNoise, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# 1. KONFIGURACJA
folder_name = "Wyniki_GAN"
if not os.path.exists(folder_name): os.makedirs(folder_name)

# Parametry
SEQ_LEN = 64            
PREDICT_STEPS = 2000    
EPOCHS = 150            
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

    # Pobranie danych (kolumny 1 i 2)
    signal_raw = df.iloc[::DOWNSAMPLE, 1:3].values
    
except FileNotFoundError:
    print("Błąd: Brak pliku foo.csv")
    exit()

# Skalowanie
scaler = MinMaxScaler(feature_range=(-1, 1))
signal_scaled = scaler.fit_transform(signal_raw)

X, y = [], []
for i in range(len(signal_scaled) - SEQ_LEN):
    X.append(signal_scaled[i : i + SEQ_LEN])
    y.append(signal_scaled[i + SEQ_LEN])

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)

if len(X) < BATCH_SIZE: BATCH_SIZE = len(X)

# 3. MODELE
def build_generator():
    inp = Input(shape=(SEQ_LEN, 2))
    x = GaussianNoise(0.05)(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    out = Dense(2, activation='tanh')(x)
    model = Model(inp, out)
    return model

def build_discriminator():
    inp_seq = Input(shape=(SEQ_LEN, 2))
    inp_next = Input(shape=(2,))
    x = LSTM(64)(inp_seq)
    x = LeakyReLU(negative_slope=0.2)(x)
    combined = Concatenate()([x, inp_next])
    c = Dense(32)(combined)
    c = LeakyReLU(negative_slope=0.2)(c)
    out = Dense(1, activation='sigmoid')(c)
    model = Model([inp_seq, inp_next], out)
    return model

generator = build_generator()
discriminator = build_discriminator()

g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

# 4. TRENING (Hybrid Loss)
@tf.function
def train_step(real_seq, real_next):
    with tf.GradientTape() as tape_d:
        generated_next = generator(real_seq, training=True)
        pred_real = discriminator([real_seq, real_next], training=True)
        pred_fake = discriminator([real_seq, generated_next], training=True)
        d_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(pred_real), pred_real) + 
                                tf.keras.losses.binary_crossentropy(tf.zeros_like(pred_fake), pred_fake))

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    
    with tf.GradientTape() as tape_g:
        generated_next = generator(real_seq, training=True)
        pred_fake = discriminator([real_seq, generated_next], training=True)
        g_adv_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(pred_fake), pred_fake)
        g_mae_loss = tf.reduce_mean(tf.abs(real_next - generated_next))
        total_g_loss = tf.reduce_mean(g_adv_loss) + 100.0 * g_mae_loss

    grads_g = tape_g.gradient(total_g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads_g, generator.trainable_variables))
    return d_loss, total_g_loss

print("Start treningu...")
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)

for epoch in range(EPOCHS):
    d_losses, g_losses = [], []
    for seq, nxt in dataset:
        d, g = train_step(seq, nxt)
        d_losses.append(d)
        g_losses.append(g)
    if (epoch+1) % 10 == 0:
        print(f"Epoka {epoch+1}/{EPOCHS} | D: {np.mean(d_losses):.4f} | G: {np.mean(g_losses):.4f}")

# 5. GENEROWANIE
print("Generowanie przebiegu...")
curr_seq = X[-1].reshape(1, SEQ_LEN, 2) 
history_seq = curr_seq.copy()           

preds = []
for i in range(PREDICT_STEPS):
    next_point = generator.predict(curr_seq, verbose=0)
    preds.append(next_point[0])
    curr_seq = np.concatenate([curr_seq[:, 1:, :], next_point.reshape(1, 1, 2)], axis=1)
    if i % 100 == 0: print(f"Krok {i}/{PREDICT_STEPS}")

preds = np.array(preds)
preds_inv = scaler.inverse_transform(preds)
hist_inv = scaler.inverse_transform(history_seq[0])

ts = datetime.datetime.now().strftime("%H%M%S")

# 6. WIZUALIZACJA

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
plt.title("Prognoza GAN")
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

plt.title("Przebieg czasowy sygnałów")
plt.xlabel("Czas [ms]")
plt.ylabel("Amplituda [-1, 1]")

plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(folder_name, f"Sygnal_{ts}.png"))
plt.pause(3)
plt.close()

print(f"\nGotowe! Wyniki zapisane w folderze: {folder_name}")
