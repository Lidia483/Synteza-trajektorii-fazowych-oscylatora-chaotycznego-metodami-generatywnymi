import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- KONFIGURACJA MODELU ---
# Długość sekwencji wejściowej. Ile poprzednich punktów wykorzystujemy do prognozy.
SEQUENCE_LENGTH = 50
# Ile punktów danych trenujemy (np. do 20000)
TRAINING_POINTS = 20000
TIMESTEP = 2.5e-6 # Zgodnie z raw2csv.py

## 1. Wczytanie i przygotowanie danych
# Zakładamy, że plik foo.csv został wygenerowany przez poprawiony raw2csv.py (z V(v1) i V(v2))
try:
    # Używamy header=None, jeśli plik foo.csv nie ma nagłówków
    data = pd.read_csv('foo.csv', header=None)
    # Wybieramy tylko dane sygnału (kolumny 1 i 2: ChL i ChR)
    # Kolumna 0 to czas
    signal_data = data.iloc[:TRAINING_POINTS, 1:3].values
except FileNotFoundError:
    print("BŁĄD: Nie znaleziono pliku 'foo.csv'. Upewnij się, że uruchomiłeś poprawiony raw2csv.py.")
    exit()

# Wizualizacja pierwszych 500 punktów sygnału (przed preprocessingiem)
plt.figure(figsize=(12, 5))
# Poprawione lw i nazwy śladów
plt.plot(signal_data[:500, 0], label='Channel Left (V(v1))', lw=1.5)
plt.plot(signal_data[:500, 1], label='Channel Right (V(v2))', lw=1.5)
plt.title('Fragment Sygnału Chaotycznego (Normalizacja [-1, 1])')
# Poprawiony opis osi X
plt.xlabel(f't')
plt.ylabel('Amplituda (Normalizowana)')
plt.legend()
plt.grid(True)
plt.show()

## 2. Tworzenie sekwencji czasowych dla LSTM
# Funkcja do tworzenia par wejście-wyjście (X, y) dla szeregów czasowych
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # X: fragment sygnału o długości seq_length
        X.append(data[i:i + seq_length])
        # y: następny punkt po sekwencji X
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(signal_data, SEQUENCE_LENGTH)

# Sprawdzenie kształtów (X ma kształt [próbki, kroki czasowe, cechy])
print(f"Kształt danych wejściowych (X): {X.shape}")
print(f"Kształt danych wyjściowych (y): {y.shape}")

## 3. Podział na zbiór treningowy i testowy
# Zbiór testowy użyjemy do sprawdzenia jak model radzi sobie z prognozowaniem
# na danych, których nie widział.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

## 4. Budowa Modelu LSTM
model = Sequential([
    # Warstwa LSTM: 100 jednostek, zwraca sekwencje
    LSTM(100, return_sequences=True, input_shape=(SEQUENCE_LENGTH, signal_data.shape[1])),
    Dropout(0.2), # Zapobieganie przetrenowaniu
    # Druga warstwa LSTM: 100 jednostek, nie zwraca sekwencji
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    # Warstwa wyjściowa: Dwie jednostki (dla ChL i ChR)
    Dense(signal_data.shape[1])
])

# Kompilacja modelu (MSE dla szeregów czasowych, Adam jako optymalizator)
model.compile(optimizer='adam', loss='mse')
model.summary()

## 5. Trenowanie Modelu
print("\n--- Rozpoczynanie Treningu Modelu LSTM ---")
history = model.fit(
    X_train, y_train,
    epochs=10,        # Zwiększ tę wartość dla lepszej dokładności
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
print("--- Trening Zakończony ---")

## 6. Prognozowanie i Wizualizacja Wyników

# Użyjemy ostatniej sekwencji ze zbioru treningowego jako punkt startowy do prognozowania
# "w przyszłość".
first_batch = X_train[-1].reshape(1, SEQUENCE_LENGTH, signal_data.shape[1])
current_sequence = first_batch

# Liczba kroków do prognozowania
PREDICTION_STEPS = 500

predictions = []
for _ in range(PREDICTION_STEPS):
    # 1. Prognozowanie następnego punktu
    next_point = model.predict(current_sequence, verbose=0)[0]
    predictions.append(next_point)

    # 2. Aktualizacja sekwencji: usuń pierwszy punkt, dodaj prognozowany
    next_sequence = np.append(current_sequence[:, 1:, :], [next_point.reshape(1, -1)], axis=1)
    current_sequence = next_sequence

predictions = np.array(predictions)

# Wizualizacja (OSTATECZNE POPRAWIONE FORMATOWANIE)
plt.figure(figsize=(15, 6))

# Oryginalne dane: Ostatni fragment zbioru treningowego (Historia)
# V(v1) - Kolumna 0
plt.plot(np.arange(SEQUENCE_LENGTH), X_train[-1][:, 0], 
         'b-', 
         label='V(v1) [ChL] - Otrzymany przebieg', 
         lw=1.5)
# V(v2) - Kolumna 1
plt.plot(np.arange(SEQUENCE_LENGTH), X_train[-1][:, 1], 
         'g-', 
         label='V(v2) [ChR] - Otrzymany przebieg', 
         lw=1.5)

# Prognozowane dane: "Przyszłość"
# V(v1) - Kolumna 0
plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH + PREDICTION_STEPS), predictions[:, 0], 
         'r--', 
         label='V(v1) [ChL] - Prognoza przebiegu', 
         lw=1.5)
# V(v2) - Kolumna 1
plt.plot(np.arange(SEQUENCE_LENGTH, SEQUENCE_LENGTH + PREDICTION_STEPS), predictions[:, 1], 
         'm--', 
         label='V(v2) [ChR] - Prognoza przyebiegu', 
         lw=1.5)

# Pionowa linia oddzielająca historię od prognozy
plt.axvline(x=SEQUENCE_LENGTH, color='k', linestyle=':', label='Start prognozy')

# Opisy wykresu
plt.title(f'Prognozowanie Sygnału Chaotycznego za Pomocą LSTM (Prognoza: {PREDICTION_STEPS} Kroków)')
# TIMESTEP jest zdefiniowany jako 2.5e-6,500×2.5×10 −6 s/krok=0.00125 s 
plt.xlabel(f'Krok Czasowy (1 Krok = {2.5e-6} s)') 
plt.ylabel('Amplituda (Normalizowana)')
plt.legend()
plt.grid(True)
plt.show()
