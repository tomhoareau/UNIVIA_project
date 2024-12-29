# Prérequis pour la reproductibilité :
# ---------------------------------
# Packages requis :
# - playwright : Pour l'automatisation du navigateur et le scraping web.
#   Installez-le avec :
#       pip install playwright
#       playwright install
#
# - csv : Bibliothèque Python intégrée pour écrire des fichiers CSV.
# - time et random : Bibliothèques Python intégrées pour les délais et la simulation de comportements humains.
# - urllib.parse : Intégré pour générer des URLs encodées pour les requêtes API.
#
# Jetons API :
# - Vous avez besoin de jetons API Crawlbase valides pour le rendu normal et JavaScript.
#   Remplacez les espaces réservés ci-dessous par vos jetons.
#
# ---------------------------------

import csv  # Pour écrire les résultats extraits dans un fichier CSV
import time  # Pour ajouter des délais entre les requêtes
import random  # Pour randomiser les délais et imiter un comportement humain
from urllib.parse import urlencode  # Pour encoder les paramètres de la requête API
from playwright.sync_api import sync_playwright  # Playwright pour l'automatisation du navigateur

# Jetons API : 
normal_token = ""        
javascript_token = ""

# Fonction pour générer l'URL Crawlbase avec les paramètres nécessaires
def get_crawlbase_url(target_url, use_javascript=False):
    """
    Génère une URL d'API Crawlbase avec le jeton approprié pour le rendu.
    Args:
        target_url (str): L'URL de la page web à scraper.
        use_javascript (bool): Si le rendu JavaScript est nécessaire.
    Returns:
        str: L'URL complète pour l'API Crawlbase.
    """
    token = javascript_token if use_javascript else normal_token
    params = {
        "token": token,
        "url": target_url,
    }
    return f"https://api.crawlbase.com/?{urlencode(params)}"

# URL de base pour les résultats de recherche des universités sur Niche
base_url = "https://www.niche.com/colleges/search/best-colleges/"

# Pages de début et de fin à scraper (inclusives)
start_page = 1
end_page = 10
all_universities = []  # Liste pour stocker tous les noms d'universités

# Initialisation de Playwright pour l'automatisation du navigateur
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # Lancer le navigateur en mode headless
    page = browser.new_page()  # Ouvrir une nouvelle page dans le navigateur

    # Bloquer les ressources inutiles (images, feuilles de style, polices) pour accélérer le scraping
    page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font"] else route.continue_())

    # Boucle sur chaque page dans la plage spécifiée
    for page_number in range(start_page, end_page + 1):
        # Construire l'URL de la page pour la pagination sur Niche
        page_url = f"{base_url}?page={page_number}"
        # Générer l'URL API Crawlbase avec le rendu JavaScript
        scrape_url = get_crawlbase_url(page_url, use_javascript=True)

        print(f"Scraping page {page_number}: {scrape_url}")
        
        # Naviguer vers l'URL avec un délai d'attente de 60 secondes
        page.goto(scrape_url, timeout=60000)

        # Attendre que les titres des résultats universitaires se chargent
        try:
            page.wait_for_selector('[data-testid="search-result__title"]', timeout=60000)
        except:
            print(f"Pas de résultats ou délai de chargement dépassé pour la page {page_number}, passage à la page suivante.")
            time.sleep(random.uniform(3, 6))  # Délai aléatoire avant de réessayer
            continue

        # Localiser tous les éléments de titre d'université
        elements = page.locator('h2[data-testid="search-result__title"]')
        count = elements.count()  # Compter le nombre d'éléments trouvés

        if count == 0:
            print(f"Aucun résultat trouvé sur la page {page_number}.")
        else:
            # Extraire le texte de chaque titre d'université, le transformer et le stocker
            for i in range(count):
                university_name = elements.nth(i).inner_text().strip()
                transformed_name = university_name.lower().replace(" ", "-")  # Transformation pour uniformité
                all_universities.append(transformed_name)

        # Ajouter un délai aléatoire entre les scrapes de pages pour imiter un comportement humain
        time.sleep(random.uniform(3, 6))

    # Fermer le navigateur après le scraping
    browser.close()

# Écrire les résultats extraits dans un fichier CSV
output_file = "universites.csv"
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["university"])  # Écrire l'en-tête
    for uni in all_universities:
        writer.writerow([uni])  # Écrire chaque nom d'université

print(f"Extraction terminée. Résultats enregistrés dans '{output_file}'.")
