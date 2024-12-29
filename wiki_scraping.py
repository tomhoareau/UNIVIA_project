# wiki_scraping.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random

def get_wikipedia_page(university_name):
    """
    Récupère le titre de la page Wikipédia correspondant à `university_name`,
    ou None si aucune page n'a été trouvée.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': university_name,
        'format': 'json'
    }
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = response.json()
        if data['query']['search']:
            page_title = data['query']['search'][0]['title']
            return page_title
        else:
            return None
    except Exception as e:
        print(f"Erreur lors de la recherche de {university_name}: {e}")
        return None

def scrape_university_info(university_name):
    """
    Scrape les infos (Undergraduates, Motto, Motto in English, Website)
    pour une université `university_name` via la page Wikipédia correspondante.
    """
    # Obtenir le titre de la page Wikipedia
    page_title = get_wikipedia_page(university_name)
    if not page_title:
        # Aucune page
        return {
            'Undergraduates': '',
            'Motto': '',
            'Motto in English': '',
            'Website': ''
        }
    
    formatted_name = page_title.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{formatted_name}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; UniversityDataScraper/1.0; +http://yourwebsite.com/bot)'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion pour {university_name}: {e}")
        return {
            'Undergraduates': '',
            'Motto': '',
            'Motto in English': '',
            'Website': ''
        }
    
    # Vérifier le status code
    if response.status_code != 200:
        print(f"Échec de la récupération de {university_name}: Code {response.status_code}")
        return {
            'Undergraduates': '',
            'Motto': '',
            'Motto in English': '',
            'Website': ''
        }

    soup = BeautifulSoup(response.content, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})
    data = {}

    if infobox:
        rows = infobox.find_all('tr')
        for row in rows:
            header = row.find('th')
            if header:
                header_text = header.text.strip()
                value_cell = row.find('td')
                value_text = value_cell.text.strip() if value_cell else ''
                data[header_text] = value_text
        
        # Récupérer les champs souhaités
        undergraduates = data.get('Undergraduates', '')
        motto = data.get('Motto', '')
        motto_en = data.get('Motto in English', '')
        
        # Extraire l'URL du site officiel
        website = ''
        for a in infobox.find_all('a', href=True):
            if 'Official website' in a.text or 'Website' in a.text:
                website = a['href']
                break

        if not website:
            # Sinon on essaie le champ 'Website'
            website = data.get('Website', '')

        return {
            'Undergraduates': undergraduates,
            'Motto': motto,
            'Motto in English': motto_en,
            'Website': website
        }
    else:
        print(f"Infobox non trouvée pour {university_name}")
        return {
            'Undergraduates': '',
            'Motto': '',
            'Motto in English': '',
            'Website': ''
        }

def scrape_universities(df, name_col='Name', sleep_min=0.1, sleep_max=0.2):
    """
    Pour chaque université dans `df[name_col]`, récupère les infos Wikipédia
    (Undergraduates, Motto, Motto in English, Website) et 
    les ajoute au DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant au moins la colonne `name_col`.
        name_col (str): Nom de la colonne contenant le nom des universités.
        sleep_min (float): Délai minimum entre deux requêtes (en secondes).
        sleep_max (float): Délai maximum entre deux requêtes (en secondes).

    Returns:
        pd.DataFrame: Le DataFrame original, enrichi des colonnes :
            - Undergraduates
            - Motto
            - Motto in English
            - Website
    """
    undergraduates_list = []
    motto_list = []
    motto_en_list = []
    website_list = []

    # Itération sur la colonne name_col avec progress bar
    for name in tqdm(df[name_col], desc="Scraping Wikipedia"):
        info = scrape_university_info(name)
        undergraduates_list.append(info['Undergraduates'])
        motto_list.append(info['Motto'])
        motto_en_list.append(info['Motto in English'])
        website_list.append(info['Website'])

        # Pause aléatoire pour éviter de surcharger le site
        time.sleep(random.uniform(sleep_min, sleep_max))
    
    # Ajouter les nouvelles colonnes
    df['Undergraduates'] = undergraduates_list
    df['Motto'] = motto_list
    df['Motto in English'] = motto_en_list
    df['Website'] = website_list

    return df




def clean_undergraduates(value):
    # Convertir en chaîne pour éviter les erreurs si c'est déjà un int/float/None
    val_str = str(value)
    # Couper avant la 1ère parenthèse
    val_str = val_str.split('(')[0]
    # Supprimer tout ce qui n'est pas chiffre (ex : virgules, espaces, etc.)
    val_str = re.sub(r'\D', '', val_str)
    # Convertir en entier si possible
    return int(val_str) if val_str.isdigit() else None

import re

def remove_parentheses_and_brackets(text):
    """Supprime tout ce qui est entre parenthèses ou crochets, puis retire les espaces en trop."""
    text = str(text)
    # Supprime les parenthèses (et leur contenu)
    text = re.sub(r'\(.*?\)', '', text)
    # Supprime les crochets [ ... ]
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()




