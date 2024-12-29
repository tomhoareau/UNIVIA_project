# <span style="color:#008080">1. Accroche</span>

> *« Le nombre de lycéens francophones partant étudier à l’étranger après le bac a été **multiplié par quatre** depuis 2020. »*

Cette statistique illustre clairement l’**intérêt grandissant** pour les études internationales. Or, si l'attrait pour les prestigieuses universités américaines est de plus en plus fort, les obstacles restent nombreux : **coûts de scolarité** élevés, **procédures d’admission complexes**, **choix des programmes** parfois déroutants… et les informations manquent souvent de clarté.

**Comment simplifier et démocratiser l’accès à ces opportunités internationales ?** C’est pour répondre à ce défi que nous lançons le projet **Univia**, une plateforme interactive conçue pour éclairer et accompagner chaque candidat dans ses démarches.

---

# <span style="color:#008080">2. Introduction et Contexte</span>

**Univia** vise à centraliser toutes les informations utiles aux futurs étudiants internationaux, de la sélection des programmes jusqu’aux formalités administratives. Au cœur de cette démarche : l’exploitation de la **donnée** pour mieux comprendre et mieux **recommander**.

<span style="color:#2E8B57">**Ce présent projet**</span> constitue **le premier volet** de l’initiative Univia. Nous allons y collecter et analyser un **ensemble de données** relatives aux universités américaines :  
- Taux d’acceptation  
- Scores requis (SAT, ACT)  
- Coûts de scolarité nets  
- Programmes et filières populaires  

Notre ambition est double :  
1. **Mieux décrypter la sélectivité** et les spécificités de chaque établissement, afin de fournir des informations claires et vérifiées.  
2. **Poser les bases d’un algorithme de recommandation** permettant de suggérer des universités adaptées au profil et aux objectifs de chaque élève.

---

# <span style="color:#008080">3. Pourquoi la Data et l’Analyse sont Cruciales</span>

Dans un contexte où l’information se révèle **fragmentée** (articles de presse, sites officiels, blogs étudiants), **l’analyse de données** s’impose comme un outil puissant pour :

- **Identifier** les tendances et les variables clés (ex. coût, notes requises, programmes les plus demandés).  
- **Comparer** objectivement les établissements en termes de sélectivité, d’aides financières, ou d’atouts académiques.  
- **Orienter** les étudiants vers les meilleures opportunités selon leur profil scolaire, financier et personnel.  

En offrant une **vision agrégée et claire** de toutes ces composantes, nous pouvons déjà lever certains freins à la mobilité. Cette **transparence** est fondamentale pour que chaque élève puisse envisager un projet d’études à l’étranger **en toute connaissance de cause**.

---

# <span style="color:#008080">4. Structure du Projet</span>

Au fil de ce notebook, nous allons :

1. **Collecter** et présenter nos premières données sur les universités américaines *(frais de scolarité, scores SAT/ACT, etc.)*.  
2. **Nettoyer et analyser** ces informations pour mettre en évidence les insights majeurs *(taux d’acceptation, filières phares, répartition géographique, etc.)*.  
3. **Prototyper** un <span style="color:#2E8B57">système de recommandation initial</span> qui, à terme, saura guider chaque utilisateur vers les établissements les plus en phase avec son profil.  

Ce travail servira de **fondation** pour les développements futurs d’Univia, qui s’étendront bientôt aux universités d’autres pays et intégreront de nouvelles fonctionnalités (préparation aux tests, accompagnement administratif, etc.).

---

# <span style="color:#008080">5. Collecte et Préparation des Données</span>

### 5.1 Sources et Fiabilité
Pour constituer notre **base de données initiale**, nous avons choisi de **scraper** les informations de [niche.com](https://www.niche.com/), un site reconnu pour la qualité et l’actualité de ses informations sur les universités. La fiabilité de Niche repose sur la quantité de retours d’étudiants, d’anciens élèves et de parents, ce qui en fait une **source de référence** pour dresser un panorama assez précis des établissements américains.

### 5.2 Méthodologie
Afin d’éviter d’être bloqués par le site et de capturer efficacement les données, nous avons utilisé **Playwright** via deux scripts :  
- **get_university_names.py**  
- **get_universities_info.py**  

Playwright nécessite une **installation préalable** pour automatiser un navigateur et réaliser le scraping sans se faire « flagger » comme un bot.

### 5.3 Enrichissement via ChatGPT
Pour **compléter** et **classifier** certains champs (par exemple le classement global d’une université), nous avons envoyé les informations collectées à l’API de ChatGPT grâce au script **get_ranking_gpt.py**.  
- Cette approche **accélère** le travail de recherche  
- La **qualité de la donnée** n’est pas parfaite, mais reste satisfaisante pour un prototype *(esprit Lean Startup)*

### 5.4 Résultat du Pipeline
Les étapes de scraping (via Playwright) et de classification (via ChatGPT) étant **longues en exécution**, nous les avons externalisées en dehors du notebook. Elles aboutissent à deux fichiers clés :  
1. **universities_data.xlsx** : après le scraping sur Niche  
2. **universities_data_with_rankings.xlsx** : après enrichissement via ChatGPT  

Ces deux fichiers servent d’**entrée** à notre notebook, où l’analyse et le **scraping complémentaire** (notamment sur Wikipédia) sont réalisés.

---

# <span style="color:#008080">6. Comment Lancer le Projet</span>

1. **Cloner le dépôt Git** sur votre environnement (ex. [SSPCloud](https://www.sspcloud.fr/)).  

2. **Installer les dépendances** :  
   - Dans la première cellule du notebook, exécuter :  
     ```bash
     pip install -r requirements.txt
     ```
   - *NB :* Si vous voulez exécuter les scripts de scraping, installez et configurez également **Playwright**.

3. **Lancer le notebook** :
   - Selectionner l'interpreteur **python 3.10.12**
   - Ouvrez le notebook principal et exécutez toutes les cellules.  
   - Vous pourrez ainsi explorer les données, effectuer le nettoyage et constater les premiers résultats de l’analyse et du système de recommandation.

---

# <span style="color:#008080">7. Utilisation de GitHub</span>

Pour ce premier sprint, nous nous sommes **concentrés** sur la collecte et le traitement des données. Le travail étant assez segmenté, il y a eu peu de collaboration simultanée sur le même code.  
- Nous avons toutefois réalisé **quelques commits** pour illustrer notre capacité à utiliser GitHub (création de branches, documentation, etc.).  
- À l’avenir, nous prévoyons d’adopter un **workflow plus collaboratif** afin d’intégrer et de déployer rapidement de nouvelles fonctionnalités.


