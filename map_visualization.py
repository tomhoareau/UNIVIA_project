# map_creation.py

import folium
from folium.plugins import MarkerCluster
from IPython.display import display, clear_output
import ipywidgets as widgets
from jinja2 import Template
from jinja2 import Template as JinjaTemplate
import pandas as pd

def get_climate_color(state_code, state_climate, climate_colors):
    """
    Récupère la couleur associée au climat de l'État `state_code`.
    Si l'État n'est pas dans `state_climate`, renvoie 'gray'.
    """
    climate = state_climate.get(state_code)
    return climate_colors.get(climate, 'gray')

# Code HTML de la légende (tel que défini dans votre snippet)
LEGEND_HTML = '''
{% macro html(this, kwargs) %}

<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 150px;
    height: auto;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    ">
    <b>Climate Categories</b><br>
    {% for category, color in climate_colors.items() %}
        <i style="background:{{color}};width:10px;height:10px;display:inline-block;"></i>
        &nbsp;{{category}}<br>
    {% endfor %}
</div>

{% endmacro %}
'''

def create_map_widget(df_for_map, state_climate, climate_colors, state_geo):
    """
    Crée et renvoie trois objets:
      1) Le slider pour la note minimale,
      2) Le SelectMultiple pour le climat,
      3) Un Output widget contenant la carte interactive.

    Paramètres:
        df_for_map (pd.DataFrame): DataFrame filtrable (avec colonnes 'Latitude', 'Longitude', 'low_grade_adjusted', 'Climate', etc.)
        state_climate (dict): Dictionnaire {state_code -> climate_category}
        climate_colors (dict): Dictionnaire {climate_category -> couleur_hex}
        state_geo (str): URL ou chemin local vers un GeoJSON décrivant les États US.

    Retourne:
        (IntSlider, SelectMultiple, Output): Les widgets et l'Output pour l’affichage dans le notebook.
    """

    # --- Création des widgets de filtre ---
    grade_slider = widgets.IntSlider(
        value=1300,  # Valeur initiale
        min=int(df_for_map['low_grade_adjusted'].min()),
        max=int(df_for_map['low_grade_adjusted'].max()),
        step=1,
        description='Min grade <='
    )

    # Toutes les catégories de climat disponibles
    climate_options = df_for_map['Climate'].dropna().unique().tolist()
    climate_dropdown = widgets.SelectMultiple(
        options=climate_options,
        value=climate_options,  # Sélectionne tout par défaut
        description='Climate:',
    )

    # Widget de sortie (pour la carte)
    map_output = widgets.Output()

    # Fonction de mise à jour de la carte
    def update_map(change):
        with map_output:
            clear_output()

            # Filtrer le DataFrame selon les valeurs des widgets
            filtered_df = df_for_map[
                (df_for_map['low_grade_adjusted'] <= grade_slider.value) &
                (df_for_map['Climate'].isin(climate_dropdown.value))
            ]

            # Créer la carte
            my_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # centre sur US

            # Ajouter le GeoJson (États US) avec coloration en fonction du climat
            folium.GeoJson(
                state_geo,
                name='States',
                style_function=lambda feature: {
                    'fillColor': get_climate_color(feature['id'], state_climate, climate_colors),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.6,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['State:'],
                    localize=True
                )
            ).add_to(my_map)

            # Ajouter les marqueurs (MarkerCluster) pour les universités
            marker_cluster = MarkerCluster().add_to(my_map)
            for _, row in filtered_df.iterrows():
                popup_info = (
                    f"<div style='font-family: Arial; font-size: 14px;'>"
                    f"<b style='font-size: 16px; color: #2b7a78;'>{row['Name']}</b><br>"
                    f"<b>City:</b> {row['City']}<br>"
                    f"<b>Grade low (/20):</b> {row['low_grade_adjusted']}<br>"
                    f"<b>Grade high (/20):</b> {row['high_grade_adjusted']}<br>"
                    f"<b>Acceptance Rate:</b> {row['Acceptance Rate']}<br>"
                    f"<b>Most popular major:</b> {row['major_top1']}<br>"
                    f"<b>Net Price:</b> {row['Net Price']}<br>"
                    f"</div>"
                )

                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_info, max_width=300),
                    icon=folium.Icon(color='black', icon="university", prefix="fa"),
                ).add_to(marker_cluster)

            # Ajouter la légende via MacroElement
            from folium.features import MacroElement
            legend = MacroElement()
            legend._template = JinjaTemplate(LEGEND_HTML)
            # Passer le dict climate_colors à la template
            legend._template.globals['climate_colors'] = climate_colors
            my_map.get_root().add_child(legend)

            # Afficher la carte
            display(my_map)

    # Observer les changements
    grade_slider.observe(update_map, names='value')
    climate_dropdown.observe(update_map, names='value')

    # Carte initiale
    with map_output:
        update_map(None)

    # Renvoyer les 3 widgets (ou plus) pour les afficher dans le notebook
    return grade_slider, climate_dropdown, map_output
