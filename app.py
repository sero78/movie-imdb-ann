import streamlit as st
import numpy as np
from joblib import load

from sklearn.neural_network import MLPRegressor
import pandas as pd
import ast

# MODELİ tekrar eğit (basit demo için)
df = pd.read_csv("tmdb_5000_movies.csv")

df = df[['budget','popularity','runtime','vote_count','vote_average']].dropna()

X = df.drop('vote_average', axis=1)
y = df['vote_average']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)
model.fit(X, y)

st.title("🎬 IMDb Puan Tahmin Sistemi")

budget = st.number_input("Budget", value=100000000)
popularity = st.number_input("Popularity", value=50.0)
runtime = st.number_input("Runtime", value=120)
vote_count = st.number_input("Vote Count", value=1000)

if st.button("Tahmin Et"):
    x = np.array([[budget, popularity, runtime, vote_count]])
    x = scaler.transform(x)
    pred = model.predict(x)[0]

    st.success(f"Tahmini IMDb Puanı: {pred:.2f}")