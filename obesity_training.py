import math
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

NUM_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CAT_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad",
]

data_num = data[NUM_COLS]
data_cat = data[CAT_COLS]

# Normalizar dados numéricos e salvar o normalizador
scaler = MinMaxScaler()
normalizer = scaler.fit(data_num)
pickle.dump(normalizer, open("normalizer_obesity.pkl", "wb"))

data_num_norm = pd.DataFrame(normalizer.transform(data_num), columns=NUM_COLS)
data_cat_norm = pd.get_dummies(data_cat, prefix_sep="||", dtype=int)

data_norm = data_num_norm.join(data_cat_norm, how="left")

# Salvar colunas para uso na inferência e descrição dos centroides
pickle.dump(list(data_norm.columns), open("columns_obesity.pkl", "wb"))

# Método do cotovelo
distorcoes = []
K = range(1, 16)

for k in K:
    model = KMeans(n_clusters=k, random_state=42).fit(data_norm)
    distorcoes.append(
        sum(np.min(cdist(data_norm, model.cluster_centers_, "euclidean"), axis=1) / data_norm.shape[0])
    )

# Determinar o número ótimo de clusters pela distância geométrica à reta
x0, y0 = K[0], distorcoes[0]
xn, yn = K[-1], distorcoes[-1]

distancias = []
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print("Número ótimo de clusters:", numero_clusters_otimo)

# Treinar e salvar o modelo final
cluster_obesity = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(data_norm)
pickle.dump(cluster_obesity, open("cluster_obesity.pkl", "wb"))