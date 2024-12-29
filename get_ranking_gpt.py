import pandas as pd
from openai import OpenAI

# Initialisation du client OpenAI avec une clé API
client = OpenAI(api_key="")

# Lecture des données universitaires à partir d'un fichier Excel
df = pd.read_excel('universities_data.xlsx')

# Ajout d'une colonne "Ranking" pour stocker les classements
df["Ranking"] = None

# Boucle sur chaque ligne du DataFrame
for idx, row in df.iterrows():
    # Récupération du nom de l'université pour générer une invite
    university_name = row["Name"]
    prompt = f"Give me the 2023 USNews US national ranking of {university_name}, the answer must contain only the number and nothing else. For liberal art college the answer must contain the number then 'LAC'."

    try:
        # Envoi de la requête à l'API OpenAI pour obtenir le classement
        response = client.chat.completions.create(
            model="gpt-4o",
            store=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extraction de la réponse retournée par l'API
        answer = response.choices[0].message.content.strip()

        # Ajout de la réponse dans la colonne "Ranking"
        df.at[idx, "Ranking"] = answer

    except Exception as e:
        # Gestion des erreurs en cas d'échec de la requête
        print(f"Failed to retrieve ranking for {university_name}: {e}")
        df.at[idx, "Ranking"] = None  # Ou une autre méthode pour gérer les erreurs

# Création d'une colonne "LAC" pour indiquer les "Liberal Art Colleges"
df["LAC"] = df["Ranking"].apply(lambda x: 1 if isinstance(x, str) and "LAC" in x else 0)

# Exportation des données enrichies dans un nouveau fichier Excel
df.to_excel("universities_data_with_rankings_test.xlsx", index=False)
