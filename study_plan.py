import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime, timedelta

# Bibliothèque pour créer le fichier ICS
from ics import Calendar, Event

#
# 1) Fonctions de calcul utilitaires
#
def calculate_sat_time(target_score):
    base_hours = 30
    additional_hours = max(0, (target_score - 1200) * 0.8)
    marginal_factor = 0.95
    return base_hours + additional_hours * marginal_factor

def create_detailed_plan(total_hours, sat_hours, essay_hours, specific_essay_hours,
                         availability, start_date, max_weeks=20):
    tasks = []
    current_date = start_date
    remaining_hours = total_hours
    remaining_sat = sat_hours
    remaining_common = essay_hours
    remaining_specific = specific_essay_hours

    end_date = start_date + timedelta(weeks=max_weeks)

    while remaining_hours > 0 and current_date < end_date:
        day_name = current_date.strftime('%A')
        if day_name in availability:
            hours_today = min(availability[day_name], remaining_hours)
            task_today = []

            # 1) SAT prep
            if remaining_sat > 0 and hours_today > 0:
                sat_today = min(hours_today, remaining_sat)
                task_today.append(f"SAT Prep: {sat_today:.1f}h")
                hours_today -= sat_today
                remaining_sat -= sat_today

            # 2) Common Essay
            if remaining_common > 0 and hours_today > 0:
                essay_today = min(hours_today, remaining_common)
                task_today.append(f"Common App Essay: {essay_today:.1f}h")
                hours_today -= essay_today
                remaining_common -= essay_today

            # 3) Specific Essays
            if remaining_specific > 0 and hours_today > 0:
                specific_today = min(hours_today, remaining_specific)
                task_today.append(f"Specific Essays: {specific_today:.1f}h")
                hours_today -= specific_today
                remaining_specific -= specific_today

            used_hours = sum([float(t.split(': ')[1][:-1]) for t in task_today])
            if used_hours > 0:
                tasks.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Day': day_name,
                    'Tasks': ' | '.join(task_today),
                    'Total Hours': used_hours
                })
            remaining_hours -= used_hours

        current_date += timedelta(days=1)

    plan_df = pd.DataFrame(tasks)
    return plan_df

#
# 2) Nouvelle fonction pour exporter le plan au format ICS
#
def export_plan_to_ics(plan_df, filename="study_plan.ics"):
    """
    Génère un fichier ICS (Calendar) à partir du DataFrame `plan_df`.
    Chaque ligne du DataFrame devient un événement dans l'agenda.
    - 'Date' (YYYY-MM-DD) => début de l'événement
    - 'Tasks' => titre ou description
    - 'Total Hours' => sert à calculer l'heure de fin (on part de 09:00).
    """
    if plan_df.empty:
        return None  # Aucun événement à créer
    
    c = Calendar()

    for _, row in plan_df.iterrows():
        date_str = row['Date']  # ex: "2024-05-20"
        total_hours = row['Total Hours']
        tasks_str = row['Tasks']

        # On part du principe qu'on commence à 9h00 chaque jour
        try:
            start_dt = datetime.strptime(date_str, "%Y-%m-%d")
            start_dt = start_dt.replace(hour=9, minute=0)
            end_dt = start_dt + timedelta(hours=total_hours)
        except ValueError:
            # Si jamais la date est mal formée ou autre
            continue

        e = Event()
        e.name = f"Study Plan: {tasks_str}"
        e.begin = start_dt
        e.end = end_dt
        e.description = f"Tasks: {tasks_str}\nPlanned Hours: {total_hours}"
        c.events.add(e)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(c)
    
    return filename

#
# 3) Fonction principale pour l'interface
#
def create_study_planner(df):
    """
    Interface interactive qui :
     - Montre un SelectMultiple pour choisir des universités
     - Permet de régler la disponibilité
     - Permet de fixer un max de semaines
     - Génère un UNIQUE plan de révision (commun)
     - Affiche côte à côte la liste et le tableau
     - Ajoute un bouton "Export to Calendar" pour créer un ICS
    """
    df_local = df.copy()
    df_local.dropna(subset=['low_sat', 'high_sat'], inplace=True)
    
    df_local['Target SAT Score'] = ((df_local['low_sat'] + df_local['high_sat']) / 2).astype(int)
    df_local['SAT Prep Time (hours)'] = df_local['Target SAT Score'].apply(calculate_sat_time)
    df_local['Common App Essay Time (hours)'] = 14
    df_local['Specific Essays Time (hours)'] = 14
    df_local['Total Time (hours)'] = (
        df_local['SAT Prep Time (hours)']
        + df_local['Common App Essay Time (hours)']
        + df_local['Specific Essays Time (hours)']
    )

    # SelectMultiple
    univ_names = df_local['Name'].unique().tolist()
    univ_select = widgets.SelectMultiple(
        options=univ_names,
        description='Universities',
        layout=widgets.Layout(width='60%', height='200px'),
        style={'description_width': 'initial'}
    )

    # Sliders
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    availability_sliders = {
        day: widgets.IntSlider(value=2, min=0, max=8, description=day, continuous_update=False)
        for day in day_names
    }
    availability_label = widgets.HTML("<h4>Nombre d'heures disponibles</h4>")
    availability_box = widgets.VBox([availability_label] + list(availability_sliders.values()))
    
    # max_weeks
    max_weeks_slider = widgets.IntSlider(
        value=20,
        min=1,
        max=52,
        description='Max Weeks',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    # Boutons
    generate_button = widgets.Button(
        description="Generate Study Plan",
        button_style='success'
    )
    export_button = widgets.Button(
        description="Export to Calendar",
        button_style='info'
    )

    # Sorties
    selected_univs_output = widgets.Output()
    plan_output = widgets.Output()
    results_box = widgets.HBox([selected_univs_output, plan_output])

    # Variables internes
    # on stockera le plan généré pour pouvoir l'exporter ensuite
    generated_plan_df = pd.DataFrame()

    # Callback du bouton "Generate Study Plan"
    def on_generate_clicked(_):
        nonlocal generated_plan_df
        selected_univs_output.clear_output()
        plan_output.clear_output()
        
        selected = list(univ_select.value)
        if not selected:
            with selected_univs_output:
                display(widgets.HTML("<h4>Aucune université sélectionnée</h4>"))
            generated_plan_df = pd.DataFrame()  # vide
            return

        with selected_univs_output:
            display(widgets.HTML("<h4>Universités sélectionnées</h4>"))
            display(pd.DataFrame({'Name': selected}))

        weekly_avail = {day: availability_sliders[day].value for day in day_names}
        max_weeks = max_weeks_slider.value

        # Addition des heures pour toutes les univs sélectionnées
        subset = df_local[df_local['Name'].isin(selected)]
        total_needed = subset['Total Time (hours)'].sum()
        sat_sum = subset['SAT Prep Time (hours)'].sum()
        comm_essay_sum = subset['Common App Essay Time (hours)'].sum()
        spec_essay_sum = subset['Specific Essays Time (hours)'].sum()

        plan_df = create_detailed_plan(
            total_hours=total_needed,
            sat_hours=sat_sum,
            essay_hours=comm_essay_sum,
            specific_essay_hours=spec_essay_sum,
            availability=weekly_avail,
            start_date=datetime.now(),
            max_weeks=max_weeks
        )

        generated_plan_df = plan_df.copy()  # on stocke pour export

        with plan_output:
            display(widgets.HTML("<h4>Study Plan (commun à tous)</h4>"))
            if plan_df.empty:
                display(widgets.HTML("<p>Impossible de tout planifier ou 0 heures totales.</p>"))
            else:
                display(plan_df)

    generate_button.on_click(on_generate_clicked)

    # Callback du bouton "Export to Calendar"
    def on_export_clicked(_):
        # On utilise le DataFrame stocké (generated_plan_df)
        if generated_plan_df.empty:
            with plan_output:
                clear_output(wait=True)
                display(widgets.HTML("<p>Aucun plan disponible à exporter. Veuillez générer le plan d'abord.</p>"))
            return

        filename = "study_plan.ics"
        result = export_plan_to_ics(generated_plan_df, filename)
        with plan_output:
            clear_output(wait=True)
            if result is None:
                display(widgets.HTML("<p>Le DataFrame est vide, rien à exporter.</p>"))
            else:
                display(widgets.HTML(
                    f"<p>Fichier ICS généré: <b>{filename}</b><br>"
                    "Importez ce fichier dans votre Google Calendar (via 'Importer Calendrier').</p>"
                ))
                display(generated_plan_df)

    export_button.on_click(on_export_clicked)

    # Mise en forme finale
    right_box = widgets.VBox([availability_box, max_weeks_slider])
    top_box = widgets.HBox([univ_select, right_box])
    
    ui = widgets.VBox([
        top_box,
        widgets.HBox([generate_button, export_button]),
        results_box
    ])
    
    display(ui)
