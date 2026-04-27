import pickle
from collections import defaultdict

import pandas as pd

NUM_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Carregar modelos salvos
cluster_obesity = pickle.load(open("cluster_obesity.pkl", "rb"))
normalizer = pickle.load(open("normalizer_obesity.pkl", "rb"))
colunas = pickle.load(open("columns_obesity.pkl", "rb"))

# Converter centroides em dataframe
centroides = pd.DataFrame(cluster_obesity.cluster_centers_, columns=colunas)

# Separar colunas numéricas e categóricas
dados_num_norm = centroides[NUM_COLS]
dados_cat_norm = centroides.drop(columns=NUM_COLS)

# Desnormalizar numéricos
dados_num = pd.DataFrame(normalizer.inverse_transform(dados_num_norm), columns=NUM_COLS)

# Desnormalizar categóricos — agrupa as dummies por coluna original e pega o maior valor
col_groups = defaultdict(list)
for col in dados_cat_norm.columns:
    original = col.split("||")[0]
    col_groups[original].append(col)

dados_cat = pd.DataFrame({
    original: dados_cat_norm[cols].idxmax(axis=1).str.split("||", regex=False).str[1]
    for original, cols in col_groups.items()
})

# Descrever os segmentos
centroides_desc = dados_num.join(dados_cat)
centroides_desc.index.name = "cluster"
print(centroides_desc.to_string())