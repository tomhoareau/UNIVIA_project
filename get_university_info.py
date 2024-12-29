import pandas as pd
from playwright.sync_api import sync_playwright
import time
from urllib.parse import urlencode
import random

# Lire le fichier CSV contenant les noms des universités.
df = pd.read_csv('universites.csv')

# Nettoyer la colonne 'university' en remplaçant les traits d'union multiples par un seul
# et en supprimant les traits d'union en début et fin de chaîne.
df['university'] = df['university'].str.replace(r'-+', '-', regex=True).str.strip('-')

# Convertir les noms d'universités nettoyés en une liste pour itération.
university_names_list = df['university'].tolist()

# Jetons pour deux modes différents de Crawlbase : navigation normale et avec JavaScript activé.
normal_token = ""
javascript_token = ""

# Fonction pour générer une URL pour l'API Crawlbase en fonction des paramètres.
def get_crawlbase_url(target_url, use_javascript=False):
    """
    Génère une URL d'API Crawlbase avec le bon jeton pour le mode choisi.
    Args:
        target_url (str): L'URL de la page cible à scraper.
        use_javascript (bool): Indique si le rendu JavaScript est requis.
    Returns:
        str: L'URL d'API complète.
    """
    token = javascript_token if use_javascript else normal_token
    params = {
        "token": token,
        "url": target_url,
    }
    return f"https://api.crawlbase.com/?{urlencode(params)}"

# URL de base pour les pages des universités sur Niche.com.
base_url = "https://www.niche.com/colleges/"

# Liste pour collecter toutes les données extraites.
data = []

# Indice de départ et de fin pour les universités à scraper.
start_index = 0
end_index = 15

# Validation de la plage d'indices.
if start_index < 0 or end_index > len(university_names_list) or start_index >= end_index:
    raise ValueError("Plage d'indices invalide.")

# Utilisation de Playwright pour l'automatisation de la navigation.
with sync_playwright() as p:
    # Lancer un navigateur Chromium en mode headless.
    browser = p.chromium.launch(headless=True)
    
    # Créer un contexte de navigateur avec JavaScript désactivé pour accélérer le scraping.
    context = browser.new_context(java_script_enabled=False)
    page = context.new_page()

    # Bloquer les ressources inutiles (images, styles) pour réduire le temps de chargement.
    page.route("**/*", lambda route: route.abort() 
               if route.request.resource_type in ["image", "stylesheet", "font"] else route.continue_())

    # Itérer sur la tranche spécifiée de la liste des universités.
    for index, university_name in enumerate(university_names_list[start_index:end_index]):  
        university_url = f"{base_url}{university_name}/"
       
        # Générer l'URL proxy Crawlbase pour la page actuelle de l'université.
        scraper_url = get_crawlbase_url(university_url, use_javascript=False)

        print(f"Scraping URL: {scraper_url}")

        try:
            # Naviguer vers la page via Crawlbase avec un délai d'attente.
            page.goto(scraper_url, wait_until="domcontentloaded", timeout=60000)

            # Vérifier si l'élément principal existe pour confirmer le chargement des données.
            if page.locator('.MuiGrid-root.MuiGrid-container.nss-1skb4mj').count() == 0:
                print("L'élément principal n'est pas visible. Activer JavaScript pourrait aider.")

            # Extraction du nom de l'université.
            name = None
            if page.locator('h1').count() > 0:
                name_text = page.locator('h1').first.text_content(timeout=5000)
                if name_text:
                    name = name_text.split('\n')[0].strip()

            # Extraction de l'adresse et de la ville.
            address, city = None, None
            if page.locator('.profile__address--compact').count() > 0:
                address_full = page.locator('.profile__address--compact').first.text_content(timeout=5000)
                if address_full:
                    address_parts = address_full.split('\n')
                    address = address_parts[0].strip() if len(address_parts) > 0 else None
                    city = address_parts[1].strip() if len(address_parts) > 1 else None

            # Extraction des données scalaires (e.g., classement, prix net).
            scalar_data = {}
            scalar_items = page.locator('.MuiGrid-root.MuiGrid-container.nss-1skb4mj .scalar')
            for i in range(scalar_items.count()):
                try:
                    label = scalar_items.nth(i).locator('.scalar__label').text_content(timeout=5000)
                    value = scalar_items.nth(i).locator('.scalar__value').text_content(timeout=5000)
                    if label: 
                        label = label.strip()
                    if value: 
                        value = value.split('\xa0')[0].strip()
                    if label and value:
                        scalar_data[label] = value
                except Exception as e:
                    print(f"Erreur lors de l'extraction du champ scalaire {i}: {e}")

            # Extraction des gammes SAT/ACT et du taux d'acceptation.
            sat_range, act_range, acceptance_rate = None, None, None
            if page.locator('.scalar--three').count() > 0:
                scalar_sections = page.locator('.scalar--three')
                for i in range(scalar_sections.count()):
                    try:
                        three_label = scalar_sections.nth(i).locator('.scalar__label').text_content(timeout=5000)
                        if three_label:
                            three_label = three_label.strip()
                        value_span = scalar_sections.nth(i).locator('.scalar__value span:not([class])').first
                        val = None
                        if value_span.count() > 0:
                            val = value_span.text_content(timeout=5000)
                            if val:
                                val = val.strip()

                        if three_label == "SAT Range" and sat_range is None:
                            sat_range = val
                        elif three_label == "ACT Range" and act_range is None:
                            act_range = val
                        elif three_label == "Acceptance Rate" and acceptance_rate is None:
                            acceptance_rate = val
                    except Exception as e:
                        print(f"Erreur lors de l'extraction des données dans la section 'scalar--three': {e}")

            # Extraction du revenu médian 6 ans après l'obtention du diplôme.
            median_income_6_years = None
            median_earning_scalars = page.locator('.profile__bucket--1 .scalar')
            for i in range(median_earning_scalars.count()):
                try:
                    label_text = median_earning_scalars.nth(i).locator('.scalar__label').text_content(timeout=5000)
                    if label_text and "Median Earnings 6 Years After Graduation" in label_text.strip():
                        value_span = median_earning_scalars.nth(i).locator('.scalar__value span').first
                        if value_span.count() > 0:
                            value_text = value_span.text_content(timeout=5000)
                            if value_text:
                                median_income_6_years = value_text.strip()
                        break
                except Exception as e:
                    print(f"Erreur lors de l'extraction du revenu médian (index {i}): {e}")

            # Extraction des matières populaires.
            subjects = []
            subjects_list = page.locator('.popular-entities-list > li')
            for i in range(subjects_list.count()):
                try:
                    subject_name = subjects_list.nth(i).locator('.popular-entity__name').text_content(timeout=5000)
                    descriptor = subjects_list.nth(i).locator('.popular-entity-descriptor').text_content(timeout=5000)
                    descriptor_suffix = subjects_list.nth(i).locator('.popular-entity-descriptor__suffix').text_content(timeout=5000)
                    if subject_name: 
                        subject_name = subject_name.strip()
                    if descriptor: 
                        descriptor = descriptor.strip()
                    if descriptor_suffix: 
                        descriptor_suffix = descriptor_suffix.strip()
                    subjects.append(f"{subject_name} ({descriptor} {descriptor_suffix})")
                except Exception as e:
                    print(f"Erreur lors de l'extraction des matières {i}: {e}")

            # Combiner toutes les données extraites dans un dictionnaire.
            university_data = {
                "Name": name,
                "City": city,
                "Address": address,
                "SAT Range": sat_range,
                "ACT Range": act_range,
                "Acceptance Rate": acceptance_rate,
                "Median Income 6 Years After Graduation": median_income_6_years,
                "Popular Majors": ", ".join(subjects),
                **scalar_data
            }
            data.append(university_data)
            print(f"Données collectées pour {university_name}: {university_data}")

        except Exception as e:
            # En cas d'échec du scraping pour une université, consigner l'erreur et continuer.
            print(f"Erreur lors du scraping de {university_url}: {e}")
            continue

        # Attendre un délai aléatoire pour imiter un comportement humain.
        time.sleep(random.uniform(3, 6))

    # Fermer le navigateur une fois terminé.
    browser.close()

# Convertir les données collectées en un DataFrame et les enregistrer dans un fichier Excel.
output_df = pd.DataFrame(data)
output_df.to_excel(f"universities_data_{start_index}_to_{end_index}_part4.xlsx", index=False)
print(f"Données enregistrées dans 'universities_data_{start_index}_to_{end_index}_part4.xlsx'.")
