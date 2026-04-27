import pickle
from collections import defaultdict

import pandas as pd

NUM_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CAT_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
# NObeyesdad não é informado — é o que o modelo vai inferir indiretamente via cluster

# Dados do novo paciente
novo_paciente_num = pd.DataFrame(
    [[29, 1.62, 78, 2, 3, 2, 1, 1]],
    columns=NUM_COLS,
)
novo_paciente_cat = pd.DataFrame(
    [["Female", "yes", "yes", "Sometimes", "no", "no", "Sometimes", "Public_Transportation"]],
    columns=CAT_COLS,
)

# Normalizar dados numéricos com o normalizador salvo
normalizer = pickle.load(open("normalizer_obesity.pkl", "rb"))
paciente_num_norm = pd.DataFrame(normalizer.transform(novo_paciente_num), columns=NUM_COLS)

# Codificar dados categóricos com o mesmo padrão do treinamento
paciente_cat_norm = pd.get_dummies(novo_paciente_cat, prefix_sep="||", dtype=int)

# Montar dataframe com todas as colunas esperadas pelo modelo, na ordem correta
colunas = pickle.load(open("columns_obesity.pkl", "rb"))
paciente_parcial = paciente_num_norm.join(paciente_cat_norm, how="left")
paciente_normalizado = pd.concat([paciente_parcial, pd.DataFrame(columns=colunas)]).fillna(0)[colunas]

# Inferir o cluster
cluster_obesity = pickle.load(open("cluster_obesity.pkl", "rb"))
cluster_paciente = cluster_obesity.predict(paciente_normalizado)
print("Cluster do paciente:", cluster_paciente[0])

# Descrever o cluster inferido
centroides = pd.DataFrame(cluster_obesity.cluster_centers_, columns=colunas)

dados_num_norm = centroides[NUM_COLS]
dados_cat_norm = centroides.drop(columns=NUM_COLS)

dados_num = pd.DataFrame(normalizer.inverse_transform(dados_num_norm), columns=NUM_COLS)

col_groups = defaultdict(list)
for col in dados_cat_norm.columns:
    col_groups[col.split("||")[0]].append(col)

dados_cat = pd.DataFrame({
    original: dados_cat_norm[cols].idxmax(axis=1).str.split("||", regex=False).str[1]
    for original, cols in col_groups.items()
})

centroides_desc = dados_num.join(dados_cat)
centroides_desc.index.name = "cluster"
print("\nDescrição do cluster:")
print(centroides_desc.loc[[cluster_paciente[0]]].to_string())