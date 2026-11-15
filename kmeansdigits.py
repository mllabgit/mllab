import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()

X = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [2, 5, 8, 10, 12]

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    print("K =", k)
    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)
    print()

plt.figure(figsize=(8, 5))
plt.plot(k_values, [
    KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
    for k in k_values
], marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means: Elbow Method")
plt.grid()
plt.show()