#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
formulaire.py

Affiche un formulaire d'informations dans un Notebook Jupyter (via ipywidgets),
en récupérant la liste des majors depuis final_dataset.xlsx (colonnes major_top1, major_top2, major_top3).
Les réponses sont enregistrées dans 'reponses_formulaire.xlsx' en écrasant l'ancien contenu.
"""

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime
import os

def display_form(dataset_path="final_dataset.xlsx"):
    """
    Affiche un formulaire interactif et enregistre les réponses dans reponses_formulaire.xlsx.
    Chaque nouvelle soumission écrase l'ancienne version du fichier.
    
    :param dataset_path: Chemin vers le fichier Excel contenant les colonnes major_top1, major_top2, major_top3.
    """
    # ==============================================================
    # 1) RÉCUPÉRER LA LISTE DES MAJORS UNIQUES DEPUIS final_dataset.xlsx
    # ==============================================================
    try:
        df = pd.read_excel(dataset_path)
        majors_cols = ["major_top1", "major_top2", "major_top3"]
        unique_majors = set()

        for col in majors_cols:
            if col in df.columns:
                # Récupérer les valeurs non nulles et les ajouter dans un set
                col_majors = df[col].dropna().unique().tolist()
                unique_majors.update(col_majors)
        
        # Retirer d'éventuels doublons et trier
        majors_list = sorted(list(unique_majors))
    except Exception as e:
        majors_list = ["Computer Science", "Biology", "Economics", "Business", "Engineering", "Maths", "Autre"]
        print(f"⚠️  Impossible de charger le fichier {dataset_path}. Erreur : {e}")
        print("Utilisation d'une liste de majors par défaut :", majors_list)

    # ==============================================================
    # 2) DÉFINIR LES WIDGETS
    # ==============================================================
    title = widgets.HTML(
        "<h1 style='text-align: center; color: darkblue;'>Questionnaire Universitaire</h1>"
    )

    # --- Informations Générales ---
    prenom_widget = widgets.Text(
        value="Jean",
        description="Prénom :",
        layout=widgets.Layout(width='50%')
    )
    nom_widget = widgets.Text(
        value="Dupont",
        description="Nom :",
        layout=widgets.Layout(width='50%')
    )
    age_widget = widgets.IntSlider(
        value=18,
        min=14,
        max=100,
        description="Âge :",
        continuous_update=False,
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    # --- Majeure & Mineure ---
    majors_dropdown = widgets.Dropdown(
        options=majors_list,
        description="Majeure actuelle :",
        value=majors_list[0] if majors_list else "Computer Science",  # Valeur par défaut
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    minor_dropdown = widgets.Dropdown(
        options=["Aucune"] + majors_list,  # On ajoute "Aucune" au début
        description="Mineure :",
        value="Aucune",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    # --- Préférences Universitaires ---
    interest1_widget = widgets.Dropdown(
        options=majors_list,
        description="Intérêt 1 :",
        value=majors_list[1] if len(majors_list) > 1 else "Biology",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    interest2_widget = widgets.Dropdown(
        options=majors_list,
        description="Intérêt 2 :",
        value=majors_list[2] if len(majors_list) > 2 else "Economics",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    interest3_widget = widgets.Dropdown(
        options=majors_list,
        description="Intérêt 3 :",
        value=majors_list[3] if len(majors_list) > 3 else "Business",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    university_size_widget = widgets.RadioButtons(
        options=["Grande université", "Petite université", "Indifférent"],
        description="Taille de l'université :",
        value="Indifférent",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    campus_life_widget = widgets.RadioButtons(
        options=["Oui", "Non", "Indifférent"],
        description="La vie de campus est-elle importante pour vous ?",
        value="Oui",
        layout=widgets.Layout(width='70%'),
        style={'description_width': 'initial'}
    )
    location_widget = widgets.Dropdown(
        options=["Rural", "Ville moyenne", "Grande ville", "Indifférent"],
        description="Localisation :",
        value="Grande ville",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    budget_widget = widgets.IntSlider(
        value=30,
        min=5,
        max=100,
        step=1,
        description="Budget (k€) :",
        continuous_update=False,
        layout=widgets.Layout(width='60%'),
        style={'description_width': 'initial'}
    )
    sports_widget = widgets.RadioButtons(
        options=["Oui", "Non", "Indifférent"],
        description="Attrait pour le sport :",
        value="Indifférent",
        layout=widgets.Layout(width='70%'),
        style={'description_width': 'initial'}
    )

    # --- Critères Académiques ---
    gpa_widget = widgets.FloatSlider(
        value=3.0,
        min=0.0,
        max=4.0,
        step=0.1,
        description="GPA :",
        continuous_update=False,
        layout=widgets.Layout(width='60%'),
        style={'description_width': 'initial'}
    )
    sat_widget = widgets.IntText(
        value=1200,
        description="Score SAT :",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    act_widget = widgets.IntText(
        value=26,
        description="Score ACT :",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    # --- Programmes Spécifiques ---
    specific_program_widget = widgets.Text(
        value="Aucun",
        description="Programme spécifique :",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    research_interest_widget = widgets.RadioButtons(
        options=["Oui", "Non", "Indifférent"],
        description="Avez-vous un intérêt pour la recherche ?",
        value="Non",
        layout=widgets.Layout(width='70%'),
        style={'description_width': 'initial'}
    )
    diversity_importance_widget = widgets.RadioButtons(
        options=["Oui", "Non", "Indifférent"],
        description="La diversité est-elle un critère pour vous ?",
        value="Indifférent",
        layout=widgets.Layout(width='70%'),
        style={'description_width': 'initial'}
    )

    # --- Risque & Ambition ---
    risk_level_widget = widgets.IntSlider(
        value=5,
        min=0,
        max=10,
        description="Niveau de risque :",
        continuous_update=False,
        layout=widgets.Layout(width='70%'),  # Bar plus longue
        style={'description_width': 'initial'}
    )
    ambition_level_widget = widgets.IntSlider(
        value=5,
        min=0,
        max=10,
        description="Niveau d'ambition :",
        continuous_update=False,
        layout=widgets.Layout(width='70%'),  # Bar plus longue
        style={'description_width': 'initial'}
    )

    # --- Style de Recommandation ---
    diversification_widget = widgets.RadioButtons(
        options=["Très similaires", "Similaires", "Variées", "Mode découverte"],
        description="Diversification :",
        value="Similaires",
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    # Bouton de soumission
    submit_button = widgets.Button(
        description="Soumettre",
        button_style='success',
        layout=widgets.Layout(width='30%')
    )

    # ==============================================================
    # 3) FONCTION DE COLLECTE & D'ENREGISTREMENT
    # ==============================================================
    def collect_responses(_):
        """
        Récupère les réponses, les affiche et les enregistre dans 'reponses_formulaire.xlsx'
        en écrasant l'ancienne version.
        """
        responses = {
            "Prénom": prenom_widget.value,
            "Nom": nom_widget.value,
            "Âge": age_widget.value,
            "Majeure_actuelle": majors_dropdown.value,
            "Mineure": minor_dropdown.value,
            "Intérêt_1": interest1_widget.value,
            "Intérêt_2": interest2_widget.value,
            "Intérêt_3": interest3_widget.value,
            "Taille_Université": university_size_widget.value,
            "Vie_campus_importante": campus_life_widget.value,
            "Localisation": location_widget.value,
            "Budget_annuel_kEUR": budget_widget.value,
            "Attrait_sport": sports_widget.value,
            "GPA": gpa_widget.value,
            "Score_SAT": sat_widget.value,
            "Score_ACT": act_widget.value,
            "Programme_spécifique": specific_program_widget.value,
            "Interet_recherche": research_interest_widget.value,
            "Diversité_critère": diversity_importance_widget.value,
            "Niveau_risque": risk_level_widget.value,
            "Niveau_ambition": ambition_level_widget.value,
            "Diversification_propositions": diversification_widget.value,
            "Horodatage": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        clear_output()
        display(widgets.HTML("<h2 style='color: green;'>Merci pour vos réponses !</h2>"))

        df_responses = pd.DataFrame([responses])
        display(df_responses)

        # Écraser l'ancien fichier Excel et sauvegarder le nouveau
        df_responses.to_excel("reponses_formulaire.xlsx", index=False)

        print("\nLes réponses ont été enregistrées dans 'reponses_formulaire.xlsx'. (Fichier écrasé à chaque soumission)")

    # Associer l'action du bouton
    submit_button.on_click(collect_responses)

    # ==============================================================
    # 4) ORGANISATION DU FORMULAIRE
    # ==============================================================
    general_info_section = widgets.VBox([
        widgets.HTML("<h2>Informations Générales</h2>"),
        prenom_widget,
        nom_widget,
        age_widget
    ])

    major_minor_section = widgets.VBox([
        widgets.HTML("<h2>Majeure & Mineure</h2>"),
        majors_dropdown,
        minor_dropdown
    ])

    university_prefs_section = widgets.VBox([
        widgets.HTML("<h2>Préférences Universitaires</h2>"),
        interest1_widget,
        interest2_widget,
        interest3_widget,
        university_size_widget,
        campus_life_widget,
        location_widget,
        budget_widget,
        sports_widget
    ])

    academic_criteria_section = widgets.VBox([
        widgets.HTML("<h2>Critères Académiques</h2>"),
        gpa_widget,
        sat_widget,
        act_widget
    ])

    program_section = widgets.VBox([
        widgets.HTML("<h2>Programmes Spécifiques</h2>"),
        specific_program_widget,
        research_interest_widget,
        diversity_importance_widget
    ])

    risk_ambition_section = widgets.VBox([
        widgets.HTML("<h2>Risque & Ambition</h2>"),
        risk_level_widget,
        ambition_level_widget
    ])

    diversification_section = widgets.VBox([
        widgets.HTML("<h2>Style de Recommandation</h2>"),
        diversification_widget
    ])

    form_layout = widgets.VBox([
        title,
        general_info_section,
        widgets.HTML("<hr>"),
        major_minor_section,
        widgets.HTML("<hr>"),
        university_prefs_section,
        widgets.HTML("<hr>"),
        academic_criteria_section,
        widgets.HTML("<hr>"),
        program_section,
        widgets.HTML("<hr>"),
        risk_ambition_section,
        widgets.HTML("<hr>"),
        diversification_section,
        widgets.HTML("<hr>"),
        submit_button
    ])

    # Afficher le formulaire
    display(form_layout)
