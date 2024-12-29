import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.express as px
import matplotlib.pyplot as plt

def perform_clustering_analysis(
    df: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
    weights: dict,
    k: int = 4,
    show_elbow: bool = True
):
    """
    Effectue la préparation des données, l'imputation, 
    le label encoding, le clustering (K-Means) et la visualisation PCA.

    Args:
        df (pd.DataFrame): Le DataFrame initial contenant les colonnes indiquées.
        numerical_cols (list): Liste des colonnes numériques.
        categorical_cols (list): Liste des colonnes catégorielles.
        weights (dict): Dictionnaire associant chaque feature à son poids.
        k (int, optional): Nombre de clusters K-Means (par défaut, 4).
        show_elbow (bool, optional): Affiche ou non le plot Elbow method (par défaut, True).

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les composantes PCA et le cluster K-Means.
        plotly.graph_objects.Figure: La figure Plotly de la projection PCA colorée par cluster.
        pd.DataFrame: Un DataFrame des loadings PCA (poids des features par composante).
    """

    # ===============================
    # 1) Copie du DataFrame pour éviter les effets de bord
    # ===============================
    df_local = df.copy()

    # ===============================
    # 2) Imputation des valeurs manquantes
    # ===============================
    numerical_imputer = SimpleImputer(strategy='mean')
    df_local[numerical_cols] = numerical_imputer.fit_transform(df_local[numerical_cols])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_local[categorical_cols] = categorical_imputer.fit_transform(df_local[categorical_cols])

    # ===============================
    # 3) Encodage des variables catégorielles
    # ===============================
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_local[col] = le.fit_transform(df_local[col])
        label_encoders[col] = le

    # ===============================
    # 4) Sélection et standardisation des features
    # ===============================
    features = numerical_cols + categorical_cols
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_local[features])

    # ===============================
    # 5) Application des poids
    # ===============================
    weighted_features = scaled_features * np.array([weights[f] for f in features])

    # ===============================
    # 6) Méthode du coude (Elbow method) pour choisir k
    # ===============================
    if show_elbow:
        inertia = []
        K = range(1, 11)
        for k_ in K:
            kmeans_temp = KMeans(n_clusters=k_, random_state=42)
            kmeans_temp.fit(weighted_features)
            inertia.append(kmeans_temp.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    # ===============================
    # 7) PCA pour visualisation 2D
    # ===============================
    pca = PCA(n_components=2)
    components = pca.fit_transform(weighted_features)
    df_local['component_1'] = components[:, 0]
    df_local['component_2'] = components[:, 1]

    # Récupérer les loadings (poids des features dans chaque PC)
    pca_components = pca.components_
    pca_loadings = pd.DataFrame(
        pca_components.T,
        index=features,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )

    # ===============================
    # 8) K-Means final avec k clusters
    # ===============================
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_local['cluster_kmeans'] = kmeans.fit_predict(weighted_features)

    # ===============================
    # 9) Visualisation interactive (Plotly)
    # ===============================
    fig = px.scatter(
        df_local,
        x='component_1',
        y='component_2',
        color='cluster_kmeans',
        hover_name='Name',   # Doit exister dans le df
        hover_data={
            'City': True,    # Doit exister aussi
            'Ranking': True  # etc.
        },
        title='PCA Visualization of K-Means Clustering',
        labels={
            'component_1': 'Principal Component 1',
            'component_2': 'Principal Component 2',
            'cluster_kmeans': 'Cluster'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=10))

    # ===============================
    # 10) Retour des résultats
    # ===============================
    return df_local, fig, pca_loadings
