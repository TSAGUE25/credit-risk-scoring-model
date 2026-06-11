# Scoring de Risque Crédit — Probabilité de Défaut Calibrée

> **Modèle calibré, KS Statistic, Gini et analyse d'équité algorithmique**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-Finance-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

Les banques utilisent des modèles de scoring crédit pour évaluer la probabilité de défaut des emprunteurs. Ces modèles sont soumis à des exigences réglementaires strictes : calibration, explicabilité et équité algorithmique.

---

## Problème traité

20 000 dossiers simulés. Construire un modèle calibré (probabilités réalistes), mesurer la discrimination (KS, Gini), définir un seuil d'acceptation et vérifier l'absence de biais discriminatoire.

---

## Solution proposée

Feature engineering (ratio d'effort, leverage), Gradient Boosting calibré par Isotonic Regression, courbe KS, Gini = 2×AUC-1, seuil d'acceptation multi-scénarios, analyse disparate impact par groupe.

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
credit-risk-scoring-model/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/credit-risk-scoring-model.git
cd credit-risk-scoring-model

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
KS = 0.52 | Gini = 0.71 | AUC = 0.855 (simulés)
```

---

## Valeur métier

Aide à la décision d'octroi de crédit. Conformité réglementaire (RGPD, AI Act).

---

## Limites

Données simulées. Pas de validation biais légal réel. Pas de backtesting.

---

## Prochaines améliorations

Validation temporelle sur données historiques. Interface explicabilité RGPD.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

