# algo_reco.py

"""
Ce module implémente un algorithme de recommandation basique
qui lit les préférences de l'utilisateur depuis 'reponses_formulaire.xlsx'
et les données des universités depuis 'final_dataset.xlsx'.
Il renvoie une liste (DataFrame) d'environ 8 à 12 universités recommandées
accompagnées d'un score de "fitting".
"""

import pandas as pd
import numpy as np
import random

def run_recommendation(
    dataset_path="final_dataset.xlsx",
    user_responses_path="reponses_formulaire.xlsx",
    nb_min=8,
    nb_max=12
):
    """
    Lit les réponses utilisateur et le dataset d'universités, calcule un score
    de compatibilité et retourne une liste de recommandations (8 à 12 établissements).
    
    Paramètres
    ----------
    dataset_path : str
        Chemin vers le fichier Excel contenant les données des universités.
    user_responses_path : str
        Chemin vers le fichier Excel contenant les préférences de l'utilisateur.
    nb_min : int
        Nombre minimum d'universités à renvoyer.
    nb_max : int
        Nombre maximum d'universités à renvoyer.
    
    Retour
    ------
    pd.DataFrame
        Un DataFrame comprenant au moins les colonnes ['Name', 'City', 'Score_Fitting']
        représentant les recommandations.
    """

    # 1) Charger les données
    try:
        df_unis = pd.read_excel(dataset_path)
    except Exception as e:
        raise FileNotFoundError(f"Impossible de charger le dataset {dataset_path} : {e}")

    try:
        df_user = pd.read_excel(user_responses_path)
    except Exception as e:
        raise FileNotFoundError(f"Impossible de charger les réponses utilisateur {user_responses_path} : {e}")

    if df_user.empty:
        raise ValueError("Le fichier de réponses utilisateur est vide.")

    # On part du principe qu'il n'y a qu'une seule ligne de réponses (la plus récente)
    user_info = df_user.iloc[-1].to_dict()

    # 2) Récupérer les infos utiles de l'utilisateur
    user_majors = [
        user_info.get("Majeure_actuelle", None),
        user_info.get("Intérêt_1", None),
        user_info.get("Intérêt_2", None),
        user_info.get("Intérêt_3", None)
    ]
    user_majors = [m for m in user_majors if isinstance(m, str) and m.strip() != ""]

    user_budget = user_info.get("Budget_annuel_kEUR", 0) * 1000  # en dollars ?
    user_sat = user_info.get("Score_SAT", 0)
    user_mode = user_info.get("Diversification_propositions", "Similaires")
    user_risk = user_info.get("Niveau_risque", 5)
    user_ambition = user_info.get("Niveau_ambition", 5)

    # 3) Définir une fonction de scoring pour chaque université
    def compute_fitting_score(row):
        score = 0.0

        # a) Matching des majors
        #   On compte combien de majors de l'utilisateur matchent les top3 de l'université
        matches = 0
        for m in user_majors:
            if pd.notna(row.get("major_top1")) and m.lower() == str(row["major_top1"]).lower():
                matches += 1
            if pd.notna(row.get("major_top2")) and m.lower() == str(row["major_top2"]).lower():
                matches += 1
            if pd.notna(row.get("major_top3")) and m.lower() == str(row["major_top3"]).lower():
                matches += 1
        
        # Normaliser sur 4 potentiels (Majeure + 3 Intérêts) => 0 à 1
        # (option simple : matches / 4)
        # On peut ajuster si on considère la "majeure" plus importante que les "intérêts"
        score += (matches / 4.0) * 2.0  # Pondération x2 pour donner de l'importance

        # b) Budget
        net_price = row.get("Net Price", 999999)
        # Si l'utilisateur a un budget en dollars, on compare directement
        if not pd.isna(net_price):
            if user_budget == 0:
                # S'il n'a pas renseigné, on ne pénalise pas
                budget_score = 0.5
            else:
                # Plus l'université est en dessous du budget, meilleur est le score
                # ex: if net_price <= user_budget => 1, sinon on diminue
                ratio = net_price / max(1, user_budget)
                if ratio <= 1:
                    budget_score = 1.0
                elif ratio > 2:
                    budget_score = 0.0
                else:
                    # lineaire entre 1 et 2
                    # ratio 1.5 => 0.5
                    budget_score = 1.0 - (ratio - 1.0)
            score += budget_score
        else:
            # Si on n'a pas de Net Price
            score += 0.3

        # c) Score SAT
        # On compare le SAT utilisateur aux low_sat et high_sat
        low_sat = row.get("low_sat", 0)
        high_sat = row.get("high_sat", 1600)
        if pd.isna(low_sat):
            low_sat = 0
        if pd.isna(high_sat):
            high_sat = 1600

        if user_sat >= low_sat:
            # S'il est supérieur au min, c'est un signe de faisabilité
            # S'il dépasse largement high_sat => on peut considérer que c'est bon
            if user_sat > high_sat:
                sat_score = 1.0
            else:
                # Normalisons
                # ex: user_sat=low_sat => 0.6, user_sat=high_sat => 1.0
                if high_sat - low_sat == 0:
                    sat_score = 0.7
                else:
                    sat_score = 0.6 + 0.4 * ((user_sat - low_sat) / (high_sat - low_sat))
        else:
            # Trop bas => 0
            sat_score = 0.0
        score += sat_score

        # d) Facteur "découverte" ou "diversification"
        #   Si mode découverte, on ajoute un léger facteur d'aléa
        if user_mode == "Mode découverte":
            # +/- 0.2 max
            random_factor = random.uniform(-0.1, 0.2)
            score += random_factor

        return score

    # 4) Appliquer le scoring sur chaque université
    df_unis["Score_Fitting"] = df_unis.apply(compute_fitting_score, axis=1)

    # 5) Optionnel : ajustement en fonction du risque et de l'ambition
    #   Par exemple, si l'étudiant est très ambitieux (8-10), on favorise
    #   les unis plus sélectives. (Ici, un exemple simple)
    #   acceptance < 0.2 => unis très sélectives
    #   On ajoute un petit bonus
    acceptance_col = "Acceptance Rate"
    if acceptance_col in df_unis.columns:
        df_unis[acceptance_col] = df_unis[acceptance_col].fillna(1.0)  # si NaN => 1
        if user_ambition > 7:
            # bonus s'il vise la sélectivité (< 20% acceptance)
            df_unis.loc[df_unis[acceptance_col] < 0.20, "Score_Fitting"] += 0.3
        elif user_ambition < 3:
            # bonus aux unis plus accessibles
            df_unis.loc[df_unis[acceptance_col] > 0.50, "Score_Fitting"] += 0.3

    # 6) Trier par Score_Fitting descendant
    df_unis = df_unis.sort_values("Score_Fitting", ascending=False)

    # 7) Sélectionner entre nb_min et nb_max universités
    #    On prend les plus hautes, sauf si la distribution est très resserrée on peut en prendre un peu plus
    recommended_unis = df_unis.head(nb_max)

    # Optionnel : on peut ajuster le nombre exact en fonction de l'écart de scores
    # Mais ici, on s'en tient à "entre nb_min et nb_max".
    if len(recommended_unis) < nb_min:
        # Si jamais il y a peu de datas, on sort tout
        pass

    # 8) Retourner seulement quelques colonnes pour la clarté
    #    ex. Name, City, Score_Fitting, ...
    recommended_unis = recommended_unis[["Name", "City", "Score_Fitting"]].copy()

    return recommended_unis.reset_index(drop=True)
