import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, LeakyReLU, Concatenate, GaussianNoise, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class SystemGAN:
    def __init__(self):
        self.folder_name = "Wyniki_GAN"
        if not os.path.exists(self.folder_name): 
            os.makedirs(self.folder_name)

        self.seq_len = 64            
        self.predict_steps = 2000    
        self.epochs = 150            
        self.batch_size = 32         
        self.timestep = 2.5e-6       
        self.dt_eff = self.timestep

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.analizator = AnalizaChaosu(self.folder_name)
        
        self.generator = self.buduj_generator()
        self.discriminator = self.buduj_dyskryminator()
        
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

    def wczytaj_dane(self, nazwa_pliku):
        print("Wczytywanie danych...")
        try:
            df = pd.read_csv(nazwa_pliku, header=None)
            data_len = len(df)

            if df.shape[1] < 3:
                raise ValueError(f"Plik {nazwa_pliku} musi mieć co najmniej 3 kolumny.")
            
            if data_len < 5000:
                downsample = 1
                print("Mały zbiór danych, pobieranie wszystkich próbek.")
            else:
                downsample = 5
                print(f"Duży zbiór danych, pobieranie co {downsample} próbki.")

            self.signal_raw = df.iloc[::downsample, 1:3].values
            self.dt_eff = self.timestep * downsample
            
        except FileNotFoundError:
            print(f"Błąd, brak pliku {nazwa_pliku}")
            raise
        except ValueError as e:
            print(f"Błąd wczytywania danych: {e}")
            raise

        signal_scaled = self.scaler.fit_transform(self.signal_raw)

        X, y = [], []
        for i in range(len(signal_scaled) - self.seq_len):
            X.append(signal_scaled[i : i + self.seq_len])
            y.append(signal_scaled[i + self.seq_len])

        self.X = np.array(X).astype(np.float32)
        self.y = np.array(y).astype(np.float32)

        if len(self.X) < self.batch_size: 
            self.batch_size = len(self.X)

    def buduj_generator(self):
        inp = Input(shape=(self.seq_len, 2))
        x = GaussianNoise(0.05)(inp)
        
        x = Conv1D(filters=64, kernel_size=5, padding='same')(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        
        x = LSTM(128, return_sequences=True)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dropout(0.2)(x)
        
        x = LSTM(64)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        
        out = Dense(2, activation='tanh')(x)
        return Model(inp, out)

    def buduj_dyskryminator(self):
        inp_seq = Input(shape=(self.seq_len, 2))
        inp_next = Input(shape=(2,))
        x = LSTM(64)(inp_seq)
        x = LeakyReLU(negative_slope=0.2)(x)
        combined = Concatenate()([x, inp_next])
        c = Dense(32)(combined)
        c = LeakyReLU(negative_slope=0.2)(c)
        out = Dense(1, activation='sigmoid')(c)
        return Model([inp_seq, inp_next], out)

    @tf.function
    def krok_treningu(self, real_seq, real_next):
        with tf.GradientTape() as tape_d:
            generated_next = self.generator(real_seq, training=True)
            pred_real = self.discriminator([real_seq, real_next], training=True)
            pred_fake = self.discriminator([real_seq, generated_next], training=True)
            d_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(pred_real), pred_real) + 
                                    tf.keras.losses.binary_crossentropy(tf.zeros_like(pred_fake), pred_fake))

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        
        with tf.GradientTape() as tape_g:
            generated_next = self.generator(real_seq, training=True)
            pred_fake = self.discriminator([real_seq, generated_next], training=True)
            
            g_adv_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(pred_fake), pred_fake)
            g_mae_loss = tf.reduce_mean(tf.abs(real_next - generated_next))
            
            ostatni_punkt_sekwencji = real_seq[:, -1, :]
            kara_za_skoki = tf.reduce_mean(tf.square(generated_next - ostatni_punkt_sekwencji))
            
            total_g_loss = tf.reduce_mean(g_adv_loss) + 100.0 * g_mae_loss + 1.0 * kara_za_skoki

        grads_g = tape_g.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))
        return d_loss, total_g_loss

    def trenuj(self):
        print("Start treningu...")
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).shuffle(1000).batch(self.batch_size, drop_remainder=True)

        for epoch in range(self.epochs):
            d_losses, g_losses = [], []
            for seq, nxt in dataset:
                d, g = self.krok_treningu(seq, nxt)
                d_losses.append(d)
                g_losses.append(g)
            if (epoch+1) % 10 == 0:
                print(f"Epoka {epoch+1}/{self.epochs} | D {np.mean(d_losses):.4f} | G {np.mean(g_losses):.4f}")

    def generuj(self):
        print("Generowanie przebiegu...")
        ts = datetime.datetime.now().strftime("%H%M%S")
        
        idx_start = 0
        if len(self.X) > self.predict_steps:
            idx_start = len(self.X) - self.predict_steps - 1
            
        curr_seq = self.X[idx_start].reshape(1, self.seq_len, 2) 
        history_seq = curr_seq.copy()           

        preds = []
        for i in range(self.predict_steps):
            next_point = self.generator(curr_seq, training=False).numpy()
            preds.append(next_point[0])
            curr_seq = np.concatenate([curr_seq[:, 1:, :], next_point.reshape(1, 1, 2)], axis=1)
            if i % 100 == 0: 
                print(f"Krok {i}/{self.predict_steps}")

        preds = np.array(preds)
        preds_inv = self.scaler.inverse_transform(preds)
        hist_inv = self.scaler.inverse_transform(history_seq[0])

        self.analizator.przeprowadz_pelna_analize(preds_inv, self.dt_eff, ts)

        plt.figure(figsize=(15, 6))
        t_hist = np.arange(self.seq_len) * self.timestep
        t_pred = np.arange(self.seq_len, self.seq_len + len(preds_inv)) * self.timestep

        plt.plot(t_hist, hist_inv[:, 0], 'b', label='Historia V1')
        plt.plot(t_hist, hist_inv[:, 1], 'g', label='Historia V2')
        plt.plot(t_pred, preds_inv[:, 0], 'r', alpha=0.8, label='Prognoza V1')
        plt.plot(t_pred, preds_inv[:, 1], 'm', alpha=0.8, label='Prognoza V2')

        plt.title("Przebieg czasowy sygnałów")
        plt.xlabel("Czas")
        plt.ylabel("Amplituda")

        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(os.path.join(self.folder_name, f"Sygnal_{ts}.png"))
        plt.close()

        print(f"Gotowe, wyniki zapisane w folderze {self.folder_name}")



class AnalizaChaosu:
    def __init__(self, folder_name="Wyniki_GAN"):
        self.folder_name = folder_name
        self.styl_ramki = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
        if not os.path.exists(self.folder_name): 
            os.makedirs(self.folder_name)

    def calc_lyapunov_divergence(self, signal, emb_dim=3, tau=1, min_tsep=10, max_k=50):
        N = len(signal)
        N_emb = N - (emb_dim - 1) * tau
        X = np.array([signal[i : i + emb_dim*tau : tau] for i in range(N_emb)])
        
        nn_indices = []
        for i in range(N_emb):
            dists = np.max(np.abs(X - X[i]), axis=1)
            dists[max(0, i-min_tsep):min(N_emb, i+min_tsep+1)] = np.inf
            nn = np.argmin(dists)
            nn_indices.append(nn)
            
        S = np.zeros(max_k)
        c = np.zeros(max_k)
        
        for k in range(max_k):
            for i, nn in enumerate(nn_indices):
                if i + k < N_emb and nn + k < N_emb:
                    dist = np.max(np.abs(X[i+k] - X[nn+k]))
                    if dist > 0:
                        S[k] += np.log(dist)
                        c[k] += 1
                        
        with np.errstate(divide='ignore', invalid='ignore'):
            return S / c

    def przeprowadz_pelna_analize(self, wygenerowany_sygnal, dt_eff, ts):
        self.analiza_entropii_i_korelacji(wygenerowany_sygnal, ts)
        self.analizuj_lle_gan(wygenerowany_sygnal, ts)
        self.rysuj_atraktor(wygenerowany_sygnal, ts)

    def rysuj_atraktor(self, generowany, ts):
        print("Rysowanie atraktora...")
        plt.figure(figsize=(8, 6))
        plt.plot(generowany[:, 0], generowany[:, 1], color='purple', alpha=0.6, lw=0.5)
        plt.title("Atraktor Lorenza")
        plt.xlabel("V1")
        plt.ylabel("V2")
        plt.grid(True)
        plt.savefig(os.path.join(self.folder_name, f"Atraktor_{ts}.png"))
        plt.close()

    def analizuj_lle_gan(self, generowany, ts):
        print("Obliczanie największego wykładnika Lapunowa...")
        sygnal_1d = generowany[:, 0]
        divergence = self.calc_lyapunov_divergence(sygnal_1d)

        zakres_liniowy = 15
        if len(divergence) > zakres_liniowy:
            wspolczynniki = np.polyfit(np.arange(zakres_liniowy), divergence[:zakres_liniowy], 1)
            mle = wspolczynniki[0]
            wyraz_wolny = wspolczynniki[1]
            prosta_x = np.arange(zakres_liniowy)
            prosta_y = mle * prosta_x + wyraz_wolny
        else:
            wspolczynniki = np.polyfit(np.arange(len(divergence)), divergence, 1)
            mle = wspolczynniki[0]
            wyraz_wolny = wspolczynniki[1]
            prosta_x = np.arange(len(divergence))
            prosta_y = mle * prosta_x + wyraz_wolny

        plt.figure(figsize=(10, 6))
        plt.plot(divergence, color='green', lw=1.5, label='Dywergencja')
        plt.plot(prosta_x, prosta_y, color='red', linestyle='--', lw=2, label='Prosta dopasowania (LLE)')
        
        plt.title("Wykładnik Lapunowa") # Zmieniono z Lapunow na Lapunowa
        plt.xlabel("Kroki czasowe [kroki]")
        plt.ylabel("Logarytm odległości między trajektoriami [-]")
        plt.grid(True)
        plt.legend(loc='upper left')
        
        tekst_wyniku = f"Wartość LLE: {mle:.4f}"
        plt.text(0.95, 0.05, tekst_wyniku, 
                 transform=plt.gca().transAxes, 
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right', 
                 bbox=self.styl_ramki)
                 
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f"Lapunowa_{ts}.png")) # Zmieniono nazwę pliku
        plt.close()

    def analiza_entropii_i_korelacji(self, generowany, ts):
        print("Analiza entropii i korelacji...")
        rozmiar_okna = 100
        skok = 50
        liczba_koszy = 20
        
        def licz_znormalizowana_entropie(fragment):
            hist, _ = np.histogram(fragment, bins=liczba_koszy)
            p = hist / np.sum(hist)
            p = p[p > 0]
            entropia = -np.sum(p * np.log2(p))
            entropia_maksymalna = np.log2(liczba_koszy)
            return entropia / entropia_maksymalna if entropia_maksymalna > 0 else 0

        def bezpieczna_korelacja(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0
            return np.corrcoef(a, b)[0, 1]
        
        entropia_gen = []
        for i in range(0, len(generowany) - rozmiar_okna + 1, skok):
            frag_gen = generowany[i : i + rozmiar_okna, 0]
            entropia_gen.append(licz_znormalizowana_entropie(frag_gen))
            
        srednia_entropia = np.mean(entropia_gen)
            
        fig_ent, ax_ent = plt.subplots(figsize=(10, 6))
        ax_ent.plot(entropia_gen, label='Znormalizowana entropia', color='red')
        ax_ent.set_title("Entropia w przesuwnym oknie")
        ax_ent.set_xlabel("Numer okna")
        ax_ent.set_ylabel("Znormalizowana entropia [-]")
        ax_ent.legend(loc='upper left')
        ax_ent.grid(True)
        
        tekst_ent = f"Średnia entropia: {srednia_entropia:.4f}"
        ax_ent.text(0.95, 0.05, tekst_ent, transform=ax_ent.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=self.styl_ramki)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f"Entropia_{ts}.png"))
        plt.close()

        korelacje_gen_gen = []
        poprzedni_gen = generowany[0 : rozmiar_okna, 0]
        
        for i in range(skok, len(generowany) - rozmiar_okna + 1, skok):
            obecny_gen = generowany[i : i + rozmiar_okna, 0]
            korelacje_gen_gen.append(bezpieczna_korelacja(obecny_gen, poprzedni_gen))
            poprzedni_gen = obecny_gen
            
        srednia_korelacja = np.mean(korelacje_gen_gen)
            
        fig_kor, ax_kor = plt.subplots(figsize=(10, 6))
        ax_kor.plot(korelacje_gen_gen, label='Korelacja sąsiadujących okien', color='orange')
        ax_kor.set_title("Korelacje między fragmentami sygnału")
        ax_kor.set_xlabel("Numer okna")
        ax_kor.set_ylabel("Korelacja")
        ax_kor.legend(loc='upper left')
        ax_kor.grid(True)
        
        tekst_kor = f"Średnia korelacja: {srednia_korelacja:.4f}"
        ax_kor.text(0.95, 0.05, tekst_kor, transform=ax_kor.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=self.styl_ramki)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f"Korelacje_{ts}.png"))
        plt.close()

if __name__ == "__main__":
    moj_system = SystemGAN()
    moj_system.wczytaj_dane('foo.csv')
    moj_system.trenuj()
    moj_system.generuj()
