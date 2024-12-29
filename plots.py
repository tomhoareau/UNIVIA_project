#------------------------------------------------------------------------------

import plotly.express as px

def create_sat_scatter_plot(df, x_col, y_col, color_col, hover_cols, title):
    """
    Crée un scatter plot des scores SAT avec une échelle de couleur pour le taux d'acceptation.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Nom de la colonne pour l'axe X (e.g., 'low_sat').
        y_col (str): Nom de la colonne pour l'axe Y (e.g., 'high_sat').
        color_col (str): Nom de la colonne pour l'échelle de couleur (e.g., 'Acceptance Rate').
        hover_cols (dict): Colonnes supplémentaires pour les informations de survol.
        title (str): Titre du graphique.
    
    Returns:
        plotly.graph_objects.Figure: Une figure Plotly.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="Name",
        hover_data=hover_cols,
        color_continuous_scale="Viridis",
        labels={x_col: "Low SAT", y_col: "High SAT"},
        title=title
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Acceptance Rate"))
    return fig


#--------------------------------

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression

def analyze_and_plot_sat_values(df, acceptance_rate_threshold=0.2):
    """
    """
    # -- Prétraitement minimal --
    df_for_SAT_value = df.copy()
    df_for_SAT_value = df_for_SAT_value.dropna(subset=['low_sat', 'high_sat', 'Acceptance Rate'])
    df_for_SAT_value['low_sat'] = pd.to_numeric(df_for_SAT_value['low_sat'], errors='coerce')
    df_for_SAT_value['high_sat'] = pd.to_numeric(df_for_SAT_value['high_sat'], errors='coerce')
    df_for_SAT_value['Acceptance Rate'] = pd.to_numeric(df_for_SAT_value['Acceptance Rate'], errors='coerce')
    df_for_SAT_value = df_for_SAT_value.dropna(subset=['low_sat', 'high_sat', 'Acceptance Rate'])

    # -- Calcul du range SAT --
    df_for_SAT_value['sat_range'] = df_for_SAT_value['high_sat'] - df_for_SAT_value['low_sat']

    # -- Régression Linéaire pour Acceptance Rate attendu --
    X = df_for_SAT_value[['sat_range']]
    y = df_for_SAT_value['Acceptance Rate']
    model = LinearRegression()
    model.fit(X, y)
    df_for_SAT_value['expected_acceptance_rate'] = model.predict(X)

    # -- Identification des "Good Value" --
    sat_range_75th = df_for_SAT_value['sat_range'].quantile(0.75)
    df_for_SAT_value['acceptance_rate_diff'] = df_for_SAT_value['Acceptance Rate'] - df_for_SAT_value['expected_acceptance_rate']
    df_for_SAT_value['good_value'] = (
        (df_for_SAT_value['sat_range'] >= sat_range_75th) &
        (df_for_SAT_value['acceptance_rate_diff'] >= acceptance_rate_threshold)
    )

    # -- Préparation pour la visualisation --
    df_for_SAT_value['marker_symbol'] = df_for_SAT_value['good_value'].apply(lambda x: 'star' if x else 'circle')
    df_for_SAT_value['marker_size'] = df_for_SAT_value['good_value'].apply(lambda x: 15 if x else 7)

    # -- Scatter principal --
    fig = px.scatter(
        df_for_SAT_value,
        x="low_sat",
        y="high_sat",
        color="Acceptance Rate",
        symbol='good_value',
        symbol_map={True: 'star', False: 'circle'},
        size='marker_size',
        size_max=15,
        hover_name="Name",
        hover_data={
            "City": True,
            "Acceptance Rate": ":.0%",
            "Net Price": ":$,.0f",
            "major_top1": True,
            "graduates_top1": True,
            "Good Value": df_for_SAT_value['good_value']
        },
        color_continuous_scale="Viridis",
        labels={"low_sat": "Low SAT", "high_sat": "High SAT"},
        title="Low SAT vs High SAT Scores by Acceptance Rate"
    )

    # -- Mise à jour du titre du colorbar --
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Acceptance Rate"
        )
    )

    # -- Ajustements pour éviter la superposition --

    # 1) Déplacer la légende des symboles (circle/star)
    #    Ici, on la place à droite de la figure (x=1.02)
    fig.update_layout(
        legend=dict(
            x=1.02,            # Décalage horizontal (droite)
            xanchor="left",
            y=1,               # Décalage vertical (haut)
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"  # Fond semi-transparent (pour lisibilité)
        )
    )
    
    # 2) Déplacer la colorbar plus à droite encore
    #    pour qu'elle ne recouvre pas la légende
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Acceptance Rate",
            x=1.15,            # Plus à droite que la légende
            xanchor='left',
            y=0.5,
            yanchor='middle',
            len=0.5           # Hauteur relative de la colorbar
        ),
        margin=dict(r=160)    # Laisser un peu d'espace à droite
    )

    return fig



#--------------------------------

import plotly.graph_objects as go

def plot_adjusted_grades_histogram(df, low_col='low_grade_adjusted', high_col='high_grade_adjusted'):
    """
    Trace un histogramme comparant la distribution des notes ajustées basses et hautes.

    Paramètres :
        df (pd.DataFrame) : Le DataFrame contenant les données de notes.
        low_col (str) : Nom de la colonne pour les notes ajustées basses.
        high_col (str) : Nom de la colonne pour les notes ajustées hautes.

    Retourne :
        plotly.graph_objects.Figure : Une figure Plotly représentant l'histogramme
        comparant les notes ajustées basses et hautes.
    """
    # Vérifie que les colonnes nécessaires existent dans le DataFrame
    if low_col in df.columns and high_col in df.columns:
        fig = go.Figure()

        # Ajout d'un premier histogramme pour les notes ajustées basses
        fig.add_trace(
            go.Histogram(
                x=df[low_col],
                name='Low Adjusted Grade',
                marker_color='blue',
                opacity=0.75
            )
        )

        # Ajout d'un second histogramme pour les notes ajustées hautes
        fig.add_trace(
            go.Histogram(
                x=df[high_col],
                name='High Adjusted Grade',
                marker_color='orange',
                opacity=0.75
            )
        )

        # Configuration de la mise en forme de l'histogramme
        fig.update_layout(
            barmode='overlay',  # Superpose les barres
            title="Distribution des notes ajustées (Basses vs Hautes)",
            xaxis_title="Notes ajustées",
            yaxis_title="Fréquence",
            legend_title="Type de note"
        )

        return fig
    else:
        # Si l'une des colonnes est manquante, on soulève une exception
        raise ValueError(f"Les colonnes '{low_col}' et/ou '{high_col}' sont introuvables dans le DataFrame.")



#--------------------------------
    
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pcolors
import ipywidgets as widgets
from itertools import cycle
from IPython.display import display

def plot_top_majors(df, max_majors=20):
    """
    Crée un diagramme à barres horizontal interactif présentant les majeures les plus populaires.

    Arguments :
        df (pd.DataFrame) : Le DataFrame contenant les colonnes 'major_top1' et 'Name'.
        max_majors (int) : Le nombre maximal de majeures à afficher.

    Retour :
        None : Affiche le graphique et un widget de sortie pour des informations supplémentaires.
    """
    # Vérification que les colonnes nécessaires existent
    if 'major_top1' not in df.columns or 'Name' not in df.columns:
        print("Les colonnes 'major_top1' ou 'Name' sont manquantes dans le DataFrame.")
        return
    
    # Regrouper par 'major_top1' et compter le nombre d'universités pour chaque majeure
    major_counts = df.groupby('major_top1').agg({
        'Name': list,  # Liste des universités pour lesquelles cette majeure est top1
        'major_top1': 'count'  # Nombre d'universités ayant cette majeure en top1
    }).rename(columns={'major_top1': 'Number of Universities'}).reset_index()
    
    # Trier les résultats par ordre décroissant du nombre d'universités
    major_counts = major_counts.sort_values(by='Number of Universities', ascending=False)
    
    # Ne garder que les N premières majeures (défini par max_majors)
    major_counts = major_counts.head(max_majors)
    
    # Création d'une palette de couleurs
    colors = pcolors.qualitative.Plotly  # Palette par défaut de Plotly
    num_colors = len(colors)
    num_bars = len(major_counts)
    
    # Si le nombre de barres dépasse le nombre de couleurs disponibles, on réitère la palette
    if num_bars > num_colors:
        bar_colors = [color for _, color in zip(range(num_bars), cycle(colors))]
    else:
        bar_colors = colors[:num_bars]
    
    # Création d'un diagramme à barres horizontal avec Plotly
    fig = go.FigureWidget(
        data=[
            go.Bar(
                x=major_counts['Number of Universities'],
                y=major_counts['major_top1'],
                orientation='h',
                text=major_counts['Number of Universities'],
                customdata=major_counts['Name'],
                marker_color=bar_colors,
                hovertemplate='<b>%{y}</b><br>Number of Universities: %{x}<extra></extra>',
            )
        ],
        layout=go.Layout(
            title='Most Popular Top Majors Across Universities',
            xaxis_title='Number of Universities',
            yaxis_title='Major',
            yaxis=dict(
                categoryorder='total ascending',  # Trie les barres par nombre décroissant
                automargin=True  # Ajuste automatiquement la marge pour éviter de couper les libellés
            ),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            margin=dict(l=250, r=50, t=50, b=50),
        )
    )
    
    # Ajustement de la taille de la figure
    fig.update_layout(
        autosize=False,
        width=900,
        height=600,
    )
    
    # Création d'un widget de sortie pour afficher des informations supplémentaires lors d'un clic
    output = widgets.Output()

    # Définition de la fonction qui gère les événements de clic sur les barres
    def on_bar_click(trace, points, state):
        with output:
            # On efface le contenu précédent du widget
            output.clear_output()
            if points.point_inds:
                idx = points.point_inds[0]
                major = major_counts.iloc[idx]['major_top1']
                universities = major_counts.iloc[idx]['Name']
                print(f"Universities where '{major}' is the top major (Total: {len(universities)}):\n")
                for uni in universities:
                    print(f"- {uni}")
    
    # On attache le gestionnaire d'événement à la première trace
    fig.data[0].on_click(on_bar_click)
    
    # On affiche la figure et le widget de sortie
    display(fig, output)



#--------------------------------
    
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
import ipywidgets as widgets
from itertools import cycle
from IPython.display import display


def plot_application_deadlines(df):
    """
    Crée un graphique à barres interactif pour visualiser les dates limites de candidature.

    Arguments :
        df (pd.DataFrame) : Le DataFrame d'entrée contenant les colonnes 'Name', 'Application Deadline' et 'Deadline Date'.

    Retour :
        None : Affiche le graphique et un widget de sortie pour afficher des informations supplémentaires.
    """

    # Fonction interne pour catégoriser le type de date limite (date spécifique ou Rolling Admissions, etc.)
    def categorize_deadline(row):
        # Si la colonne 'Deadline Date' contient une valeur, on considère que c'est une date spécifique
        if pd.notnull(row['Deadline Date']):
            return 'Specific Date'
        else:
            # Sinon, on reprend la valeur présente dans 'Application Deadline' (ex : Rolling Admissions)
            return row['Application Deadline']

    # On ajoute une colonne 'Deadline Type' pour différencier les dates spécifiques des autres types
    df['Deadline Type'] = df.apply(categorize_deadline, axis=1)

    # On extrait le mois depuis la colonne 'Deadline Date' (qui est de type datetime) 
    # et on convertit ce mois en son nom complet (Janvier, Février, etc.)
    df['Deadline Month'] = df['Deadline Date'].dt.strftime('%B')

    # Pour les lignes où 'Deadline Month' est NaN, on les remplace par le contenu de 'Deadline Type'
    df['Deadline Month'] = df.apply(
        lambda row: row['Deadline Month'] if pd.notnull(row['Deadline Month']) else row['Deadline Type'],
        axis=1
    )

    # On regroupe les données par 'Deadline Month'
    #   - On agrège les noms de collèges sous forme de liste
    #   - On combine les valeurs de 'Application Deadline' (pour avoir la liste de deadlines uniques)
    #   - On prend la première date comme référence pour effectuer un tri ultérieur
    #   - On prend le premier 'Deadline Type' trouvé
    deadline_grouped = df.groupby('Deadline Month').agg({
        'Name': list,
        'Application Deadline': lambda x: ', '.join(x.unique()),
        'Deadline Date': 'first',  
        'Deadline Type': 'first'
    }).reset_index()

    # On renomme les colonnes résultantes pour plus de clarté
    deadline_grouped.columns = ['Deadline', 'Colleges', 'Deadlines', 'Deadline Date', 'Deadline Type']

    # On ajoute une colonne pour le nombre de collèges concernés par chaque deadline
    deadline_grouped['Number of Colleges'] = deadline_grouped['Colleges'].apply(len)

    # Cette fonction renvoie un entier correspondant au mois pour trier,
    # ou 13 pour placer les types Rolling / autres en fin de liste
    def get_sort_key(row):
        if pd.notnull(row['Deadline Date']):
            return row['Deadline Date'].month
        else:
            return 13  # Valeur plus élevée pour trier en dernier

    # On applique la fonction de tri puis on réorganise le DataFrame
    deadline_grouped['Sort Key'] = deadline_grouped.apply(get_sort_key, axis=1)
    deadline_grouped = deadline_grouped.sort_values(by='Sort Key')

    # On crée une palette de couleurs. On récupère d'abord la palette de base de Plotly
    unique_deadlines = deadline_grouped['Deadline'].unique()
    num_deadlines = len(unique_deadlines)
    colors = plotly.colors.qualitative.Plotly  # Palette Plotly par défaut

    # Si le nombre de catégories dépasse la palette de base,
    # on répète les couleurs pour éviter les erreurs
    if num_deadlines > len(colors):
        colors = [color for _, color in zip(range(num_deadlines), cycle(colors))]

    # On mappe chaque date limite à une couleur
    deadline_color_map = dict(zip(unique_deadlines, colors[:num_deadlines]))
    bar_colors = deadline_grouped['Deadline'].map(deadline_color_map)

    # On crée la figure interactive avec Plotly
    fig = go.FigureWidget(
        data=[
            go.Bar(
                x=deadline_grouped['Deadline'],
                y=deadline_grouped['Number of Colleges'],
                text=deadline_grouped['Number of Colleges'],
                customdata=deadline_grouped['Colleges'],
                hovertext=deadline_grouped['Deadlines'],
                marker_color=bar_colors,
                hovertemplate='<b>%{x}</b><br>Number of Colleges: %{y}<br>Deadlines: %{hovertext}',
            )
        ],
        layout=go.Layout(
            title='Number of Colleges by Application Deadline',
            xaxis_title='Application Deadline',
            yaxis_title='Number of Colleges',
            xaxis={'categoryorder': 'array', 'categoryarray': deadline_grouped['Deadline']},
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        )
    )

    # Création d'un widget de sortie pour afficher des informations supplémentaires lors du clic sur une barre
    output = widgets.Output()

    # Définition de la fonction qui gère les événements de clic sur les barres
    def on_bar_click(trace, points, state):
        with output:
            # On efface le contenu précédent du widget
            output.clear_output()
            if points.point_inds:
                idx = points.point_inds[0]
                deadline = deadline_grouped.iloc[idx]['Deadline']
                colleges = deadline_grouped.iloc[idx]['Colleges']
                print(f"Colleges with application deadline '{deadline}':")
                for college in colleges:
                    print(f"- {college}")

    # On attache le gestionnaire d'événement de clic à la première trace du graphique
    fig.data[0].on_click(on_bar_click)

    # On affiche le graphique et le widget de sortie
    display(fig, output)



#--------------------------------
    
# ranking_analysis.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

def analyze_ranking_vs_acceptance(df, ranking_threshold=10):
    """
    Analyse la relation entre l'Acceptance Rate et le Ranking, 
    et génère un scatter plot pour mettre en évidence les universités 
    ayant un 'good value'.

    Args:
        df (pd.DataFrame): DataFrame contenant 'Acceptance Rate' et 'Ranking'.
        ranking_threshold (int, optional): Différence de ranking minimale 
            pour considérer qu'une université est 'good_value'.
            (Ranking réel < Ranking attendu - threshold).

    Returns:
        fig (plotly.graph_objects.Figure): La figure Plotly générée.
    """
    # 1) Copie + nettoyage minimal
    df_for_ranking_value = df.copy()
    df_for_ranking_value = df_for_ranking_value.dropna(subset=['Acceptance Rate', 'Ranking'])

    # 2) Régression linéaire (Ranking ~ Acceptance Rate)
    X = df_for_ranking_value['Acceptance Rate'].values.reshape(-1, 1)
    y = df_for_ranking_value['Ranking'].values
    ranking_model = LinearRegression()
    ranking_model.fit(X, y)

    # Prédictions (Ranking attendu)
    df_for_ranking_value['expected_ranking'] = ranking_model.predict(X)

    # Points pour la "best fit line"
    acceptance_rate_fit = np.linspace(
        df_for_ranking_value['Acceptance Rate'].min(),
        df_for_ranking_value['Acceptance Rate'].max(),
        100
    ).reshape(-1, 1)
    ranking_fit = ranking_model.predict(acceptance_rate_fit)

    # 3) Identification des "Good Value"
    df_for_ranking_value['good_value'] = (
        df_for_ranking_value['Ranking'] < 
        (df_for_ranking_value['expected_ranking'] - ranking_threshold)
    )

    # 4) Création de la figure
    fig = px.scatter(
        df_for_ranking_value,
        x="Acceptance Rate",
        y="Ranking",
        color="good_value",
        hover_name="Name",
        hover_data={
            "City": True,
            "Ranking": True,
            "expected_ranking": True,
            "good_value": True,
        },
        color_discrete_map={True: "green", False: "blue"},
        labels={"Acceptance Rate": "Acceptance Rate", "Ranking": "Ranking"},
        title="Ranking vs Acceptance Rate with Good Value Highlight"
    )

    # Inverser l'axe Y (car un ranking plus bas est meilleur)
    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    # Ajouter la droite de régression
    fig.add_scatter(
        x=acceptance_rate_fit.flatten(),
        y=ranking_fit,
        mode='lines',
        name='Best Fit Line',
        line=dict(color='red', dash='dash')
    )

    return fig


#--------------------------------


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from sklearn.linear_model import LinearRegression

def plot_nan_proportions(df, columns):
    """
    Affiche un graphique en barres montrant la proportion de NaN
    pour chaque colonne listée dans `columns`.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        columns (list): Liste des noms de colonnes à examiner.
    """
    # Calculer la proportion de NaN pour chaque colonne
    nan_props = []
    for col in columns:
        if col in df.columns:
            prop = df[col].isna().mean()  # proportion de NaN
            nan_props.append((col, prop))
        else:
            # Si la colonne n'existe pas dans le DF, on met None
            nan_props.append((col, None))

    # Créer un DataFrame pour le tri et l’affichage
    nan_df = pd.DataFrame(nan_props, columns=["Column", "NaN Proportion"])
    # On met de côté les None (colonnes inexistantes)
    nan_df = nan_df.dropna(subset=["NaN Proportion"])
    # Tri par ordre décroissant de la proportion
    nan_df = nan_df.sort_values("NaN Proportion", ascending=False)

    # Créer la figure en barres
    fig = px.bar(
        nan_df,
        x="NaN Proportion",
        y="Column",
        orientation="h",
        title="Proportion de NaN par variable"
    )
    fig.update_layout(
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(autorange="reversed"),  # pour afficher la plus grande proportion en haut
        height=500,
        margin=dict(l=100, r=50, t=50, b=50),
    )
    fig.show()



#--------------------------------


def plot_income_histogram(df, income_col="Median Income 6 Years After Graduation"):
    """
    Affiche un histogramme de la variable `income_col`.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        income_col (str): Nom de la colonne contenant l'information sur le revenu médian.
    """
    if income_col not in df.columns:
        print(f"La colonne '{income_col}' n'existe pas dans le DataFrame.")
        return

    # On droppe les NaN avant de tracer
    valid_data = df[income_col].dropna()

    fig = px.histogram(
        valid_data,
        x=income_col,
        nbins=30,  # Ajuster le nombre de bins
        title=f"Histogramme de {income_col}"
    )
    fig.update_layout(
        xaxis_title=income_col,
        yaxis_title="Nombre d'universités (count)",
        margin=dict(l=50, r=50, t=50, b=50),
        height=500,
    )
    fig.show()



#--------------------------------


import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def regression_income_vs_ranking(df,
                                 income_col="Median Income 6 Years After Graduation",
                                 ranking_col="Ranking"):
    """
    Effectue une régression linéaire de 'income_col' sur 'ranking_col'.
    Affiche le R² et le coefficient de régression dans un print,
    et affiche un scatter plot + la droite de régression.
    Convertit automatiquement la colonne income_col si celle-ci 
    est au format '$xxx,xxx'.

    Args:
        df (pd.DataFrame): Le DataFrame contenant au minimum 'income_col' et 'ranking_col'.
        income_col (str): Nom de la colonne pour le revenu médian (en format style "$104,700" si besoin).
        ranking_col (str): Nom de la colonne pour le classement.
    """
    if income_col not in df.columns:
        print(f"La colonne '{income_col}' n'existe pas dans le DataFrame.")
        return
    if ranking_col not in df.columns:
        print(f"La colonne '{ranking_col}' n'existe pas dans le DataFrame.")
        return

    # Créer un DataFrame intermédiaire pour ne pas modifier l'original
    df_valid = df.dropna(subset=[income_col, ranking_col]).copy()

    if df_valid.empty:
        print("Pas de données suffisantes pour effectuer la régression (toutes les lignes sont NaN).")
        return

    # Convertir la colonne income_col en float si elle est au format style "$104,700"
    # 1) On cast en string (pour éviter les erreurs si déjà float)
    # 2) On supprime $ et les virgules
    # 3) On convertit en float
    df_valid[income_col] = (
        df_valid[income_col]
        .astype(str)
        .str.replace(r'[\$,]', '', regex=True)  # supprime $ et ,
        .astype(float)
    )

    # Idem pour ranking_col, si besoin, si ce n'est pas déjà numérique
    # df_valid[ranking_col] = pd.to_numeric(df_valid[ranking_col], errors='coerce')

    # Retrait des éventuelles lignes devenues NaN après conversion
    df_valid.dropna(subset=[income_col, ranking_col], inplace=True)

    if df_valid.empty:
        print("Après conversion, pas de données suffisantes pour effectuer la régression.")
        return

    X = df_valid[[ranking_col]].values.reshape(-1, 1)
    y = df_valid[income_col].values

    # Régression linéaire
    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_

    print(f"Régression : {income_col} ~ {ranking_col}")
    print(f"Coefficient de régression (slope) = {coef:.3f}")
    print(f"Ordonnée à l'origine (intercept) = {intercept:.3f}")
    print(f"R² = {r2:.3f}")

    # Préparer les données pour tracer la droite de régression
    x_min, x_max = X.min(), X.max()
    x_range = np.linspace(x_min, x_max, 100)
    y_pred = model.predict(x_range.reshape(-1, 1))

    # Scatter plot + line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_valid[ranking_col],
        y=df_valid[income_col],
        mode='markers',
        name='Data Points'
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f"{income_col} vs. {ranking_col}",
        xaxis_title=ranking_col,
        yaxis_title=income_col,
        height=600
    )
    fig.show()
