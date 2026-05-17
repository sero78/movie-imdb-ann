# ============================================================
# YAPAY SİNİR AĞLARI PROJESİ
# Film IMDb Puan Tahmini (MLP - Yapay Sinir Ağı)
# ============================================================

# ============================================================
# GEREKLİ KÜTÜPHANELER
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Yapay sinir ağı modeli
from sklearn.neural_network import MLPRegressor

# ============================================================
# 1. DATASET YÜKLEME
# ============================================================

# tmdb_5000_movies.csv
# aynı klasörde olmalı

df = pd.read_csv(r"tmdb_5000_movies.csv")

print("\nDATASET İLK 5 SATIR\n")
print(df.head())

# ============================================================
# 2. GEREKLİ KOLONLARI SEÇME
# ============================================================

df = df[[
    'budget',
    'popularity',
    'runtime',
    'vote_count',
    'genres',
    'release_date',
    'vote_average'
]]

# ============================================================
# 3. EKSİK VERİLERİ TEMİZLEME
# ============================================================

df = df.dropna()

# ============================================================
# 4. GENRE TEMİZLEME
# ============================================================

def extract_genre(x):
    try:
        genres = ast.literal_eval(x)
        return genres[0]['name']
    except:
        return "Unknown"

df['genres'] = df['genres'].apply(extract_genre)

# ============================================================
# 5. TARİHTEN YILI ÇIKARMA
# ============================================================

df['release_year'] = pd.to_datetime(df['release_date']).dt.year

# release_date artık gereksiz
df = df.drop('release_date', axis=1)

# ============================================================
# 6. GENRE ENCODING
# ============================================================

df = pd.get_dummies(df, columns=['genres'])

# ============================================================
# 7. X ve Y AYIRMA
# ============================================================

# hedef değişken
y = df['vote_average']

# özellikler
X = df.drop('vote_average', axis=1)

# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# 9. SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 10. YAPAY SİNİR AĞI MODELİ
# ============================================================

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# modeli eğit
model.fit(X_train, y_train)

# ============================================================
# 11. TAHMİN
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# 12. PERFORMANS ÖLÇÜMÜ
# ============================================================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\nMODEL SONUÇLARI\n")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")

# ============================================================
# 13. GERÇEK VS TAHMİN GRAFİĞİ
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label='Gerçek Puan')
plt.plot(predictions[:100], label='Tahmin Edilen Puan')

plt.title("Gerçek vs Tahmin")
plt.xlabel("Film")
plt.ylabel("IMDb Puanı")

plt.legend()

plt.show()

# ============================================================
# 14. İLK 10 TAHMİN
# ============================================================

print("\nİLK 10 TAHMİN\n")

for i in range(10):

    gerçek = y_test.values[i]
    tahmin = predictions[i]

    print(f"Gerçek: {gerçek:.1f} | Tahmin: {tahmin:.1f}")

# ============================================================
# PROJE TAMAMLANDI
# ============================================================