# 📚 SmartRecomAI — Système de recommandation intelligent

Système de recommandation de livres combinant des moteurs de machine learning classiques (filtrage collaboratif et recommandation par contenu) avec une couche d'agents IA spécialisés orchestrés autour d'un LLM, accessible via une interface conversationnelle avec mémoire de session.

## Objectif

Proposer des recommandations de livres personnalisées à partir de plusieurs signaux (similarité de titre, historique utilisateur, requête en langage naturel) tout en offrant une expérience utilisateur conversationnelle plutôt qu'un simple formulaire de filtres.

## Architecture

Le projet s'organise en trois couches :

1. **Moteurs de recommandation** (`recommender.py`, `train_model.py`)
   - **Filtrage collaboratif item-item** : k-NN cosinus sur une matrice user×item construite à partir des notes (dataset GoodBooks-10k), avec options de pondération et de normalisation.
   - **Recommandation par contenu** : vectorisation TF-IDF (titre, auteurs, tranche de note) et k-NN cosinus, avec une variante par centroïde utilisateur (moyenne des items aimés).
   - Les deux moteurs sont entraînés et sérialisés (`joblib`, `scipy.sparse`) via un script d'entraînement paramétrable en ligne de commande.

2. **Agents IA** (`agents/`)
   - Agents spécialisés construits avec **CrewAI**, chacun dédié à une tâche précise : recommandation par titre, par ID, recherche par contenu, recommandation collaborative, recommandation hybride, fiche détaillée d'un livre.
   - Chaque agent s'appuie sur un LLM (Groq) et un outil dédié qui interroge l'API de recommandation.
   - Un router dispatche la demande de l'utilisateur vers l'agent pertinent selon l'intention détectée (titre, ID, user_id, requête libre).

3. **API** (`api/`)
   - API **FastAPI** exposant les moteurs de recommandation (contenu, collaboratif, hybride) ainsi que des utilitaires (fiche détaillée, recherche fuzzy de titres, santé du service).

4. **Interface utilisateur** (`streamlit_app.py`)
   - Interface de chat **Streamlit** avec **mémoire conversationnelle de session** : un résumé glissant des derniers échanges est conservé et transmis aux agents à chaque tour, permettant des recommandations contextualisées sur plusieurs messages.

## Stack technique

- **Machine learning** : scikit-learn (k-NN, TF-IDF), scipy (matrices creuses), numpy, pandas
- **Agents IA / LLM** : CrewAI, Groq
- **API** : FastAPI
- **Interface** : Streamlit
- **Sérialisation** : joblib

## Lancer le projet

```bash
# Entraîner les moteurs de recommandation
python train_model.py --n-neighbors 50 --min-rating 3

# Lancer l'API
uvicorn api.main:app --reload

# Lancer l'interface de chat
streamlit run streamlit_app.py
```

## Structure du projet

```
SmartRecomAI/
├── agents/          # Agents CrewAI, router, outils, config LLM
├── api/             # API FastAPI (endpoints de recommandation)
├── artefacts/        # Modèles et matrices sérialisés
├── configuration/   # Fichiers de configuration
├── données brutes/  # Données GoodBooks-10k
├── data_loader.py   # Chargement des données
├── recommender.py   # Moteurs de recommandation (collaboratif + contenu)
├── train_model.py   # Script d'entraînement et de sérialisation
└── streamlit_app.py # Interface de chat avec mémoire de session
```
