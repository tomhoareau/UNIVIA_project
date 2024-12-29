import re
import pandas as pd

# Data structure
def add_deadline_column(df, deadline_col='Application Deadline', output_col='Deadline Date', year=2025):
    """
    Ajoute une colonne avec des dates de deadline correctement formatées et avec une année fixe.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        deadline_col (str): Le nom de la colonne contenant les deadlines.
        output_col (str): Le nom de la colonne où stocker les dates formatées.
        year (int): L'année à fixer pour toutes les dates.
    
    Returns:
        pd.DataFrame: Le DataFrame enrichi avec la nouvelle colonne des deadlines.
    """
    # Assurez-vous que la colonne est de type chaîne
    df[deadline_col] = df[deadline_col].astype(str)

    # Fonction pour analyser les deadlines et fixer l'année
    def parse_deadline(deadline):
        try:
            # Essayer de parser sans l'année
            date = pd.to_datetime(deadline, format='%B %d')
            return date.replace(year=year)
        except ValueError:
            try:
                # Essayer de parser avec une année
                date = pd.to_datetime(deadline)
                return date.replace(year=year)
            except ValueError:
                # Retourne NaT si l'analyse échoue
                return pd.NaT

    # Appliquez la fonction pour créer la colonne de deadline
    df[output_col] = df[deadline_col].apply(parse_deadline)
    
    return df

def remove_empty_rows(df):
    """
    Supprime les lignes entièrement vides d'un DataFrame.
    Une ligne est considérée vide si toutes ses colonnes sont NaN ou vides.
    """
    df = df.replace('', None)  # Remplace les chaînes vides par None
    df = df.dropna(subset=['Name'])  # Supprime les lignes où la colonne 'Name' est NaN ou None
    return df

def extract_sat_range(sat_range_str):
    """
    Extrait low_sat et high_sat à partir d'une chaîne de forme 'xxx-yyy'.
    Renvoie un pd.Series avec {'low_sat': xxx, 'high_sat': yyy}.
    """
    if pd.isna(sat_range_str):
        return pd.Series({'low_sat': None, 'high_sat': None})
    else:
        match = re.match(r'(\d+)-(\d+)', sat_range_str)
        if match:
            low_sat = int(match.group(1))
            high_sat = int(match.group(2))
            return pd.Series({'low_sat': low_sat, 'high_sat': high_sat})
        else:
            return pd.Series({'low_sat': None, 'high_sat': None})

def apply_extract_sat_range(df):
    """
    Applique la fonction `extract_sat_range` sur la colonne 'SAT Range'
    et ajoute les colonnes 'low_sat' et 'high_sat' au DataFrame.
    """
    df[['low_sat', 'high_sat']] = df['SAT Range'].apply(extract_sat_range)
    return df

def extract_majors(popular_majors_str):
    """
    Extrait la liste des majeures (major_name, graduates) à partir
    d'une chaîne de la forme 'MajorName (xx Graduates), ...'.
    Retourne une liste triée par nombre de diplômés décroissant.
    """
    if pd.isna(popular_majors_str):
        return []
    else:
        majors_list = popular_majors_str.split(', ')
        majors_data = []
        for major_info in majors_list:
            match = re.match(r'(.+?)\s*\((\d+)\s+Graduates', major_info)
            if match:
                major_name = match.group(1)
                graduates = int(match.group(2))
                majors_data.append((major_name, graduates))
        majors_data.sort(key=lambda x: x[1], reverse=True)
        return majors_data

def apply_extract_majors(df):
    """
    Applique `extract_majors` sur la colonne 'Popular Majors' pour créer 
    trois colonnes : 'major_top1', 'major_top2', 'major_top3' et 
    'graduates_top1', 'graduates_top2', 'graduates_top3'.
    Supprime la colonne 'majors_list' une fois les colonnes créées.
    """
    df['majors_list'] = df['Popular Majors'].apply(extract_majors)
    for i in range(3):
        df[f'major_top{i+1}'] = df['majors_list'].apply(lambda x: x[i][0] if len(x) > i else None)
        df[f'graduates_top{i+1}'] = df['majors_list'].apply(lambda x: x[i][1] if len(x) > i else None)
    df = df.drop(columns=['majors_list'])
    return df



#--------------------------------
# Cleaning GPT returns

# data_cleaning.py
import re

def clean_ranking(row):
    """
    Nettoie la colonne 'Ranking' selon des règles prédéfinies.
    Si LAC est 1, renvoie None. Sinon, extrait les informations de classement.
    """
    name = str(row['Name'])
    lac = str(row['LAC'])
    ranking_str = str(row['Ranking']).strip()

    # Si LAC, ranking est toujours None
    if lac == '1':
        return None

    # Définir les patterns stricts :
    # 1. Numéro unique avec éventuellement un préfixe '#' ou '$'
    single_number_pattern = r'^[#\$]?(\d+)$'
    # 2. Intervalle avec éventuellement un préfixe '#' ou '$'
    interval_pattern = r'^[#\$]?(\d+)-\d+$'
    
    # Vérifie le format de numéro unique
    single_match = re.match(single_number_pattern, ranking_str)
    if single_match:
        return int(single_match.group(1))
    
    # Vérifie le format d'intervalle
    interval_match = re.match(interval_pattern, ranking_str)
    if interval_match:
        return int(interval_match.group(1))
    
    # Si aucun des formats ne correspond, retourne None
    return None


#--------------------------------

# Add coordinates (for the map)

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


def add_coordinates_to_dataframe(df, address_col='Address', city_col='City', output_file=None):
    """
    Ajoute des coordonnées géographiques (latitude et longitude) au DataFrame basé sur les colonnes 'Address' et 'City'.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes d'adresse et de ville.
        address_col (str): Nom de la colonne contenant l'adresse.
        city_col (str): Nom de la colonne contenant la ville.
        output_file (str, optional): Chemin du fichier Excel où sauvegarder les résultats. Si None, ne sauvegarde pas.
    
    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les colonnes 'Latitude' et 'Longitude'.
    """
    # Initialiser le géocodeur
    geolocator = Nominatim(user_agent="university_locator")

    # Fonction pour obtenir les coordonnées
    def get_coordinates(full_address):
        try:
            location = geolocator.geocode(full_address, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except GeocoderTimedOut:
            return None, None

    # Créer une colonne 'Full Address' combinant adresse et ville
    df['Full Address'] = df[address_col] + ", " + df[city_col]

    # Ajouter les coordonnées au DataFrame
    df['Coordinates'] = df['Full Address'].apply(lambda x: get_coordinates(x))
    df['Latitude'] = df['Coordinates'].apply(lambda x: x[0])
    df['Longitude'] = df['Coordinates'].apply(lambda x: x[1])

    # Supprimer les colonnes intermédiaires
    df.drop(columns=['Coordinates', 'Full Address'], inplace=True)

    # Sauvegarder dans un fichier Excel si un chemin est fourni
    if output_file:
        df.to_excel(output_file, index=False)

    return df




#--------------------------------

# State and climate

import re
import pandas as pd

def add_state_and_climate(df, city_col='City'):
    """
    Ajoute les colonnes 'State' et 'Climate' au DataFrame à partir de la colonne contenant les informations de ville.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        city_col (str): Le nom de la colonne contenant les informations de ville.
    
    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les colonnes 'State' et 'Climate'.
    """
    # Fonction pour extraire l'État à partir de la colonne ville
    def extract_state(city):
        """
        Extrait l'abréviation de l'État d'une chaîne au format 'City, ST ZIPCODE'.
        """
        if pd.isna(city):  # Vérifie si la valeur est None ou NaN
            return None
        # Regex pour trouver les deux lettres majuscules (abréviation de l'État)
        match = re.search(r',\s*([A-Z]{2})\s*\d{5}', city)
        return match.group(1) if match else None

    # Dictionnaire des climats par État
    state_climate = {
        'AK': 'Cold', 'AL': 'Warm', 'AR': 'Warm', 'AZ': 'Hot', 'CA': 'Warm',
        'CO': 'Cold', 'CT': 'Temperate', 'DE': 'Temperate', 'FL': 'Hot',
        'GA': 'Warm', 'HI': 'Hot', 'IA': 'Cold', 'ID': 'Cold', 'IL': 'Cold',
        'IN': 'Cold', 'KS': 'Temperate', 'KY': 'Temperate', 'LA': 'Hot',
        'MA': 'Temperate', 'MD': 'Temperate', 'ME': 'Cold', 'MI': 'Cold',
        'MN': 'Cold', 'MO': 'Temperate', 'MS': 'Warm', 'MT': 'Cold',
        'NC': 'Temperate', 'ND': 'Cold', 'NE': 'Cold', 'NH': 'Cold',
        'NJ': 'Temperate', 'NM': 'Hot', 'NV': 'Hot', 'NY': 'Temperate',
        'OH': 'Temperate', 'OK': 'Warm', 'OR': 'Temperate', 'PA': 'Temperate',
        'RI': 'Temperate', 'SC': 'Warm', 'SD': 'Cold', 'TN': 'Temperate',
        'TX': 'Hot', 'UT': 'Dry', 'VA': 'Temperate', 'VT': 'Cold',
        'WA': 'Temperate', 'WI': 'Cold', 'WV': 'Temperate', 'WY': 'Cold',
        'DC': 'Temperate'
    }

    # Appliquer l'extraction de l'État
    df['State'] = df[city_col].apply(extract_state)

    # Mapper les climats sur la base des États
    df['Climate'] = df['State'].map(state_climate)

    return df

