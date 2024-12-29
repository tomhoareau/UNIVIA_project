import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ========== 1) Préparation & Nettoyage des données ==========

def clean_acceptance_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la colonne 'Acceptance Rate' de type string ('XX%') 
    en float entre 0 et 1 dans le DataFrame fourni.
    """
    df["Acceptance Rate"] = (
        df["Acceptance Rate"]
        .astype(str)
        .str.strip('%')
        .astype(float) 
        / 100
    )
    return df

def sat_score_to_percentile(sat_score: float) -> float:
    """
    Interpolation univariée pour associer un score SAT à un percentile.
    """
    # Mapping SAT scores to percentiles (approximate data)
    sat_scores = np.array([
        600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1550, 1600
    ])
    sat_percentiles = np.array([
        1, 1, 2, 5, 10, 16, 23, 31, 40, 49, 58, 67, 74,
        81, 88, 91, 94, 96, 98, 99, 99.9
    ])

    return np.interp(sat_score, sat_scores, sat_percentiles)

def apply_sat_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée 'low_percentile' et 'high_percentile' en appliquant sat_score_to_percentile
    aux colonnes 'low_sat' et 'high_sat'.
    """
    df['low_percentile'] = df['low_sat'].apply(sat_score_to_percentile)
    df['high_percentile'] = df['high_sat'].apply(sat_score_to_percentile)
    return df

def percentile_to_bac_grade(percentile: float) -> float:
    """
    Convertit un percentile SAT (entre 0 et 100) en note de bac (0 à 20).
    """
    # Définition des notes de bac et de leurs probabilités cumulées
    bac_grades = np.array([10, 12, 14, 16, 20])
    cumulative_probabilities = np.array([0.086, 0.415, 0.716, 0.911, 1.0])
    
    # On insère 0 pour couvrir la tranche [0, note la plus basse]
    bac_grades = np.insert(bac_grades, 0, 0)  # => [0, 10, 12, 14, 16, 20]
    cumulative_probabilities = np.insert(cumulative_probabilities, 0, 0)  # => [0, 0.086, 0.415, ...]

    # On crée la fonction d'interpolation inverse
    bac_percentile_to_grade = interp1d(
        cumulative_probabilities, bac_grades, 
        kind='linear', fill_value="extrapolate"
    )

    # Les percentiles sont supposés entre 0 et 100, on ramène ça à [0,1]
    percentile /= 100
    percentile = np.clip(percentile, 0, 1)
    return bac_percentile_to_grade(percentile)

def apply_bac_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique la conversion percentile -> note de bac 
    pour créer 'low_grade' et 'high_grade'.
    """
    df['low_grade'] = df['low_percentile'].apply(percentile_to_bac_grade)
    df['high_grade'] = df['high_percentile'].apply(percentile_to_bac_grade)
    return df

def adjust_grade_for_acceptance_rate(grade: float, acceptance_rate: float) -> float:
    """
    Ajuste la note en fonction du taux d'acceptation.
    Plus le taux d'acceptation est faible, plus la note requise est élevée.
    """
    # On peut ajuster par un facteur (1 - acceptance_rate) sur une amplitude de 1 point
    adjustment_factor = (1 - acceptance_rate)
    adjusted_grade = grade - 2 + adjustment_factor
    return min(adjusted_grade, 20)  # La note max reste 20

def apply_grade_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les colonnes 'low_grade_adjusted' et 'high_grade_adjusted'
    en ajustant la note en fonction du taux d'acceptation.
    """
    df['low_grade_adjusted'] = df.apply(
        lambda x: adjust_grade_for_acceptance_rate(x['low_grade'], x['Acceptance Rate']), axis=1
    )
    df['high_grade_adjusted'] = df.apply(
        lambda x: adjust_grade_for_acceptance_rate(x['high_grade'], x['Acceptance Rate']), axis=1
    )

    # On arrondit à 2 décimales
    df['low_grade_adjusted'] = df['low_grade_adjusted'].round(2)
    df['high_grade_adjusted'] = df['high_grade_adjusted'].round(2)
    return df

def full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chaîne toutes les étapes de préparation pour simplifier l'appel dans le notebook.
    """
    df = clean_acceptance_rate(df)
    df = apply_sat_percentiles(df)
    df = apply_bac_grades(df)
    df = apply_grade_adjustment(df)
    return df

# ========== 2) Fonctions de plotting ==========

def plot_sat_cdf():
    """
    Trace la courbe de distribution cumulative (CDF) des scores SAT.
    """
    sat_scores = np.array([
        600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1550, 1600
    ])
    sat_percentiles = np.array([
        1, 1, 2, 5, 10, 16, 23, 31, 40, 49, 58, 67, 74,
        81, 88, 91, 94, 96, 98, 99, 99.9
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(sat_scores, sat_percentiles)
    plt.title('Cumulative Distribution Function of SAT Scores')
    plt.xlabel('SAT Score')
    plt.ylabel('Percentile')
    plt.grid(True)
    plt.show()

def plot_sat_pdf():
    """
    Trace l’histogramme approx. de la densité de probabilité (PDF) des scores SAT.
    """
    sat_scores = np.array([
        600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1550, 1600
    ])
    sat_percentiles = np.array([
        1, 1, 2, 5, 10, 16, 23, 31, 40, 49, 58, 67, 74,
        81, 88, 91, 94, 96, 98, 99, 99.9
    ])

    # Calcul des milieux d'intervalles et des différences
    sat_scores_mid = (sat_scores[:-1] + sat_scores[1:]) / 2
    sat_percentiles_diff = np.diff(sat_percentiles)
    sat_scores_diff = np.diff(sat_scores)

    sat_pdf = sat_percentiles_diff / sat_scores_diff

    plt.figure(figsize=(8, 5))
    plt.bar(sat_scores_mid, sat_pdf, width=np.diff(sat_scores), edgecolor='black', alpha=0.7)
    plt.title('Approximate Probability Density Function of SAT Scores')
    plt.xlabel('SAT Score')
    plt.ylabel('Density')
    plt.grid(axis='y')
    plt.show()


# ---------------------------------------------------------------------
# 3) Fonctions pour comparer distributions SAT vs Bac
# ---------------------------------------------------------------------

def plot_sat_bac_cdf_comparison():
    """
    Compare la CDF (fonction de répartition cumulée) des scores SAT
    et la CDF (discrète) des notes de bac sur deux sous-graphiques.
    """
    # --- Données SAT (déjà utilisées dans plot_sat_cdf) ---
    sat_scores = np.array([
        600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1550, 1600
    ])
    sat_percentiles = np.array([
        1, 1, 2, 5, 10, 16, 23, 31, 40, 49, 58, 67, 74,
        81, 88, 91, 94, 96, 98, 99, 99.9
    ])

    # --- Données Bac (CDF discrète) ---
    # Bac_grades correspond aux bornes : 0, 10, 12, 14, 16, 20
    # cumulative_probabilities correspond aux valeurs cumulées : 0, 0.086, 0.415...
    bac_grades = np.array([0, 10, 12, 14, 16, 20])
    bac_cdf = np.array([0, 0.086, 0.415, 0.716, 0.911, 1.0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ======== Subplot 1 : CDF SAT ========
    axes[0].plot(sat_scores, sat_percentiles, label='SAT CDF', color='blue')
    axes[0].set_title('CDF - SAT')
    axes[0].set_xlabel('SAT Score')
    axes[0].set_ylabel('Percentile')
    axes[0].grid(True)

    # ======== Subplot 2 : CDF Bac ========
    # On peut faire un plot en escalier ou un plot simple reliant les points
    axes[1].step(bac_grades, bac_cdf, where='post', label='Bac CDF', color='green')
    axes[1].set_title('CDF - Bac')
    axes[1].set_xlabel('Bac Grade')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_sat_bac_pdf_comparison():
    """
    Compare la PDF (densité de probabilité) des scores SAT 
    et la distribution discrète des notes de bac.
    """
    # --- PDF SAT ---
    sat_scores = np.array([
        600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1050, 1100, 1150, 1200, 1250, 1300, 1350, 
        1400, 1450, 1500, 1550, 1600
    ])
    sat_percentiles = np.array([
        1, 1, 2, 5, 10, 16, 23, 31, 40, 49, 58, 67, 74,
        81, 88, 91, 94, 96, 98, 99, 99.9
    ])
    # On calcule la PDF ~ diff(cdf)/diff(scores)
    sat_scores_mid = (sat_scores[:-1] + sat_scores[1:]) / 2   # milieu de chaque intervalle
    sat_percentiles_diff = np.diff(sat_percentiles)
    sat_scores_diff = np.diff(sat_scores)
    sat_pdf = sat_percentiles_diff / sat_scores_diff

    # --- PDF Bac (discrète) ---
    bac_grades = np.array([0, 10, 12, 14, 16, 20])
    bac_cdf = np.array([0, 0.086, 0.415, 0.716, 0.911, 1.0])
    bac_pdf = np.diff(bac_cdf)
    # Pour positionner les barres, on prend les milieux des intervalles
    bac_mid = (bac_grades[:-1] + bac_grades[1:]) / 2  # ex: (0+10)/2=5, (10+12)/2=11...

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ======== Subplot 1 : PDF SAT ========
    axes[0].bar(sat_scores_mid, sat_pdf, width=np.diff(sat_scores), 
                edgecolor='black', alpha=0.7, color='blue')
    axes[0].set_title('PDF - SAT')
    axes[0].set_xlabel('SAT Score')
    axes[0].set_ylabel('Density')
    axes[0].grid(True)

    # ======== Subplot 2 : PDF Bac ========
    # Distribution discrète => on fait un bar chart directement aux milieux
    # Note : la largeur de chaque barre correspond à la distance entre bac_grades[i] et bac_grades[i+1]
    bar_widths = np.diff(bac_grades)  # ex: 10, 2, 2, 2, 4
    axes[1].bar(bac_mid, bac_pdf, width=bar_widths, 
                edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title('PDF - Bac')
    axes[1].set_xlabel('Bac Grade')
    axes[1].set_ylabel('Probability')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
