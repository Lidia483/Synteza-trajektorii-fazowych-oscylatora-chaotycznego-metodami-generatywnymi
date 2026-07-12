import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def calc_lyapunov_divergence(signal, emb_dim=3, tau=1, min_tsep=10, max_k=50):
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

def calc_rolling_sampen(signal, window=200, m=2, r_mult=0.2):
    n = len(signal)
    entropies = []
    
    for start in range(0, n - window + 1, window // 2):
        chunk = signal[start:start+window]
        r = r_mult * np.std(chunk)
        
        x_m = np.array([chunk[i:i+m] for i in range(len(chunk) - m)])
        N_m = 0
        for i in range(len(x_m)):
            dist = np.max(np.abs(x_m - x_m[i]), axis=1)
            N_m += np.sum(dist <= r) - 1
            
        x_m1 = np.array([chunk[i:i+m+1] for i in range(len(chunk) - m - 1)])
        N_m1 = 0
        for i in range(len(x_m1)):
            dist = np.max(np.abs(x_m1 - x_m1[i]), axis=1)
            N_m1 += np.sum(dist <= r) - 1
            
        if N_m <= 0 or N_m1 <= 0:
            entropies.append(0.0)
        else:
            entropies.append(-np.log(N_m1 / N_m))
            
    return entropies


class AnalizaAtraktora:
    def __init__(self, plik_csv):
        df = pd.read_csv(plik_csv, header=None)
        
        self.signal_raw = df[[1, 2]].values 
        self.predict_steps = len(self.signal_raw)
        
    def generuj_wykresy(self, ts="wynik"):
        v1 = self.signal_raw[:self.predict_steps, 0]
        v2 = self.signal_raw[:self.predict_steps, 1]

        sygnal_1d = v1 

        divergence = calc_lyapunov_divergence(sygnal_1d)
        rolling_entropy = calc_rolling_sampen(sygnal_1d, window=300)

        srednia_entropia = np.mean(rolling_entropy)
        print(f"Ostateczny wynik entropii (średnia): {srednia_entropia:.4f}")

        # Wyliczenie największego wykładnika Lapunowa i przygotowanie prostej
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
            
        print(f"Największy wykładnik Lapunowa (MLE): {mle:.4f}")

        folder_wynikowy = "atraktor"
        os.makedirs(folder_wynikowy, exist_ok=True)

# ----------------------------------------------------
        # Okno 1: Atraktor (Przestrzeń fazowa) - PROSTOKĄTNY
        # ----------------------------------------------------
        plt.figure(figsize=(8, 6)) # Lekko powiększony, wyraźnie prostokątny rozmiar
        # Ustawienie color='blue' i wyższego alpha, aby niebieski był wyraźny
        plt.plot(v1, v2, color='blue', alpha=0.8, lw=0.5) 
        plt.title("Atraktor")
        plt.xlabel("Sygnał 1")
        plt.ylabel("Sygnał 2")
        plt.grid(True)
        
        # Zmiana z 'equal' na 'auto' pozwala na narysowanie prostokąta
        plt.gca().set_aspect('auto') 
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder_wynikowy, f"Atraktor_{ts}.png"))
        plt.close()
        # ----------------------------------------------------
        # Okno 2: Wykładnik Lapunowa
        # ----------------------------------------------------
        plt.figure()
        plt.plot(divergence, 'r', lw=1.5, label='Dywergencja')
        # Rysowanie wyliczonej prostej dopasowania
        plt.plot(prosta_x, prosta_y, 'k--', lw=2, label='Prosta dopasowania (LLE)')
        
        plt.title("Wykładnik Lapunowa")
        plt.xlabel("Kroki czasowe [kroki]")
        plt.ylabel("Logarytm odległości między trajektoriami [-]")
        plt.grid(True)
        plt.legend(loc='upper left')
        
        plt.text(0.95, 0.05, f"Największy wykładnik: {mle:.4f}", 
                 transform=plt.gca().transAxes, 
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
                 
        plt.tight_layout()
        plt.savefig(os.path.join(folder_wynikowy, f"Lapunow_{ts}.png"))
        plt.close()

        # ----------------------------------------------------
        # Okno 3: Entropia próbkowa
        # ----------------------------------------------------
        plt.figure()
        plt.plot(rolling_entropy, 'g', lw=1.5, marker='o')
        plt.title("Entropia próbkowa")
        plt.xlabel("Numer analizowanego okna [kroki]")
        plt.ylabel("Wartość entropii [-]")
        plt.grid(True)
        
        plt.text(0.95, 0.05, f"Średni wynik: {srednia_entropia:.4f}", 
                 transform=plt.gca().transAxes, 
                 fontsize=11, verticalalignment='bottom', horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout()
        plt.savefig(os.path.join(folder_wynikowy, f"Entropia_{ts}.png"))
        plt.close()

        print(f"Zakończono. Wykresy zapisano w folderze: {folder_wynikowy}")

if __name__ == "__main__":
    analiza = AnalizaAtraktora("foo.csv")
    analiza.generuj_wykresy()
