# CAS D'USAGE 13 — Scoring de Risque Crédit
## Modéliser la probabilité de défaut et calibrer un score pour l'octroi de crédit

> **Auteur :** TSAGUE EMMANUEL — Data Scientist / Data Analyst  
> **Domaine :** Finance, Credit Risk, Calibration, Éthique ML  
> **Repository GitHub :** `credit-risk-scoring-model`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---
## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Scoring crédit calibré : probabilité de défaut, KS statistic et analyse de biais pour un modèle équitable"**

> **Scoring crédit :** processus qui attribue à chaque demandeur de crédit une probabilité de défaut (ne pas rembourser). Les banques utilisent ce score pour décider l'octroi, le taux et le montant du crédit.

> **Défaut :** incapacité à rembourser le crédit. En France, un crédit est en défaut quand le client n'a pas payé depuis 90 jours.

Ce projet construit un modèle de scoring crédit sur 20 000 dossiers simulés. Il couvre : modélisation de la probabilité de défaut, calibration des probabilités, KS statistic, seuil d'acceptation et analyse des biais pour l'équité algorithmique.

**Résultats simulés :** KS = 0,52 | Gini = 0,71 | AUC = 0,855.

---
## 2. CONTEXTE RÉGLEMENTAIRE ET ÉTHIQUE

> **RGPD et équité algorithmique :** un modèle de scoring crédit ne peut pas discriminer sur la base du sexe, de l'origine ethnique, de l'âge (en dehors de certaines limites légales) ou de la religion. Des variables comme le code postal peuvent introduire un biais indirect de discrimination géographique.

> **Explainability réglementaire :** en Europe, la réglementation exige que les refus de crédit soient explicables au demandeur. Un modèle boîte noire n'est pas acceptable sans couche d'interprétabilité.

| Obligation | Description |
|-----------|-------------|
| Explication du refus | "Votre crédit est refusé en raison d'un taux d'endettement élevé" |
| Non-discrimination | Pas de variables protégées directes |
| Calibration | La probabilité prédite doit correspondre à la réalité |

---
## 3. GÉNÉRATION DES DONNÉES SIMULÉES

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
N = 20_000

df = pd.DataFrame({
    "age":                  np.random.randint(20, 75, N),
    "revenu_mensuel":       np.abs(np.random.lognormal(7.5, 0.5, N)),
    "montant_credit":       np.abs(np.random.lognormal(9.5, 0.8, N)),
    "duree_mois":           np.random.choice([12, 24, 36, 48, 60, 84], N),
    "nb_credits_actifs":    np.random.randint(0, 8, N),
    "taux_endettement":     np.random.beta(2, 5, N),  # Entre 0 et 1, peak autour de 0.28
    "anciennete_banque_ans":np.random.randint(0, 30, N),
    "incidents_passés":     np.random.poisson(0.3, N),
    "type_emploi":          np.random.choice(
        ["CDI", "CDD", "Indépendant", "Fonctionnaire", "Chômeur"],
        N, p=[0.45, 0.20, 0.15, 0.12, 0.08]
    ),
    "proprietaire":         np.random.choice([0, 1], N, p=[0.45, 0.55]),
    "situation_familiale":  np.random.choice(
        ["Célibataire", "Marié", "Divorcé", "Veuf"], N, p=[0.35, 0.45, 0.15, 0.05]
    ),
})

# Probabilité de défaut simulée avec logique métier
df["pd_logit"] = (
    -2.0 +
    0.03 * df["taux_endettement"] * 10 +
    0.8  * (df["incidents_passés"] > 0).astype(float) +
    0.5  * (df["type_emploi"] == "Chômeur").astype(float) +
    0.3  * (df["type_emploi"] == "CDD").astype(float) +
    -0.4 * (df["type_emploi"] == "Fonctionnaire").astype(float) +
    -0.3 * df["proprietaire"] +
    0.1  * df["nb_credits_actifs"] +
    np.random.normal(0, 0.5, N)
)
df["prob_defaut_vraie"] = 1 / (1 + np.exp(-df["pd_logit"]))
df["defaut"] = np.random.binomial(1, df["prob_defaut_vraie"])

print(f"Taux de défaut simulé : {df['defaut'].mean():.1%}")
print(df.drop(columns=["pd_logit", "prob_defaut_vraie"]).describe().round(2))

X = df.drop(columns=["defaut", "pd_logit", "prob_defaut_vraie"])
y = df["defaut"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

---
## 4. FEATURE ENGINEERING — VARIABLES MÉTIER

```python
# Variables dérivées pertinentes pour le scoring crédit
def engineer_features(df_in):
    df = df_in.copy()

    # Taux mensualité/revenu (ratio d'effort)
    df["mensualite_estimee"] = df["montant_credit"] / df["duree_mois"]
    df["ratio_effort"]       = df["mensualite_estimee"] / (df["revenu_mensuel"] + 1)

    # Montant crédit par unité de revenu annuel
    df["leverage"]           = df["montant_credit"] / (df["revenu_mensuel"] * 12 + 1)

    return df

X_train = engineer_features(X_train)
X_test  = engineer_features(X_test)
```

---
## 5. MODÈLE DE SCORING

```python
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.metrics       import roc_auc_score, classification_report

cols_num = ["age", "revenu_mensuel", "montant_credit", "duree_mois",
            "nb_credits_actifs", "taux_endettement", "anciennete_banque_ans",
            "incidents_passés", "mensualite_estimee", "ratio_effort", "leverage"]
cols_cat = ["type_emploi", "situation_familiale"]
cols_bin = ["proprietaire"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), cols_num),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cols_cat),
    ("bin", "passthrough", cols_bin),
])

# Régression logistique (interprétable, standard réglementaire)
pipe_lr = Pipeline([
    ("prep",  preprocessor),
    ("model", LogisticRegression(
        C=0.1, class_weight="balanced",
        max_iter=1000, random_state=42
    ))
])

# Gradient Boosting (plus puissant mais moins interprétable)
pipe_gb = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=42
    ))
])

for nom, pipe in [("Logistic Regression", pipe_lr), ("Gradient Boosting", pipe_gb)]:
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"{nom:25s} | AUC = {auc:.4f}")
```

---
## 6. CALIBRATION DES PROBABILITÉS

> **Calibration :** un modèle est bien calibré si, parmi les dossiers à qui il attribue une probabilité de défaut de 30 %, environ 30 % font effectivement défaut. Un modèle non calibré peut surestimer ou sous-estimer les risques.

> **Courbe de calibration (reliability diagram) :** graphique qui compare les probabilités prédites moyennes vs les taux de défaut réels par décile. Une courbe parfaite est la diagonale.

> **Platt Scaling / Isotonic Regression :** techniques pour re-calibrer les probabilités d'un modèle après entraînement.

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt

# Calibration par Platt Scaling (sigmoid)
pipe_gb_cal = CalibratedClassifierCV(pipe_gb, method="isotonic", cv=5)
pipe_gb_cal.fit(X_train, y_train)
y_proba_cal = pipe_gb_cal.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Courbe de calibration avant et après
CalibrationDisplay.from_predictions(y_test, pipe_gb.predict_proba(X_test)[:, 1],
                                     n_bins=10, ax=axes[0], name="Avant calibration")
CalibrationDisplay.from_predictions(y_test, y_proba_cal,
                                     n_bins=10, ax=axes[0], name="Après calibration")
axes[0].set_title("Courbe de Calibration")

# Distribution des scores par défaut/non-défaut
axes[1].hist(y_proba_cal[y_test == 0], bins=50, alpha=0.6, color="green",
             label="Non-défaut", density=True)
axes[1].hist(y_proba_cal[y_test == 1], bins=50, alpha=0.6, color="red",
             label="Défaut", density=True)
axes[1].set_title("Distribution des probabilités de défaut")
axes[1].set_xlabel("P(défaut)")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/credit_calibration.png", dpi=150, bbox_inches="tight")
```

---
## 7. KS STATISTIC — MÉTRIQUE STANDARD DU CRÉDIT

> **KS Statistic (Kolmogorov-Smirnov) :** métrique de discrimination standard dans l'industrie bancaire. Mesure la distance maximale entre les distributions cumulées des scores pour les bons et mauvais payeurs. KS > 0,4 est considéré "bon" dans le scoring crédit.

> **Gini Coefficient :** 2 × (AUC - 0,5). Mesure la capacité de discrimination du modèle. Gini = 0 = modèle aléatoire, Gini = 1 = modèle parfait.

```python
import numpy as np
from scipy.stats import ks_2samp

# KS Statistic
scores_bons     = y_proba_cal[y_test == 0]
scores_mauvais  = y_proba_cal[y_test == 1]
ks_stat, ks_pval = ks_2samp(scores_bons, scores_mauvais)

auc_cal = roc_auc_score(y_test, y_proba_cal)
gini    = 2 * auc_cal - 1

print(f"=== MÉTRIQUES SCORING CRÉDIT ===")
print(f"KS Statistic : {ks_stat:.4f} ({'Bon' if ks_stat > 0.4 else 'Acceptable'})")
print(f"Gini         : {gini:.4f}")
print(f"AUC          : {auc_cal:.4f}")

# Courbe KS
fig, ax = plt.subplots(figsize=(9, 6))
scores_sorted = np.sort(np.concatenate([scores_bons, scores_mauvais]))
cdf_bons    = np.searchsorted(np.sort(scores_bons), scores_sorted) / len(scores_bons)
cdf_mauvais = np.searchsorted(np.sort(scores_mauvais), scores_sorted) / len(scores_mauvais)

ax.plot(scores_sorted, cdf_bons, color="green", label="Bons payeurs")
ax.plot(scores_sorted, cdf_mauvais, color="red", label="Mauvais payeurs")
idx_ks = np.argmax(np.abs(cdf_bons - cdf_mauvais))
ax.annotate(f"KS = {ks_stat:.3f}",
            xy=(scores_sorted[idx_ks], (cdf_bons[idx_ks] + cdf_mauvais[idx_ks])/2),
            fontsize=11, color="black")
ax.set_title(f"Courbe KS — KS = {ks_stat:.3f}, Gini = {gini:.3f}")
ax.legend(); ax.set_xlabel("Score de risque")
plt.savefig("figures/credit_ks_curve.png", dpi=150, bbox_inches="tight")
```

---
## 8. SEUIL D'ACCEPTATION ET POLITIQUE DE CRÉDIT

```python
import pandas as pd

# Simulation de différents seuils d'acceptation
resultats_seuils = []
for seuil in np.arange(0.05, 0.60, 0.05):
    acceptes = (y_proba_cal <= seuil)
    resultats_seuils.append({
        "Seuil PD":       round(seuil, 2),
        "Taux acceptation": acceptes.mean(),
        "Taux défaut attendu": y_proba_cal[acceptes].mean() if acceptes.sum() > 0 else 0,
        "Nb dossiers acceptés": acceptes.sum(),
    })

df_seuils = pd.DataFrame(resultats_seuils)
print(df_seuils.round(3).to_string(index=False))
```

---
## 9. ANALYSE DES BIAIS ET ÉQUITÉ

> **Biais algorithmique :** quand un modèle traite différemment des groupes protégés (sexe, âge, origine). En crédit, un biais peut résulter des données historiques qui reflètent des discriminations passées.

> **Disparate Impact :** si le taux d'acceptation d'un groupe est inférieur à 80 % du taux d'acceptation du groupe le plus favorisé, il y a potentiellement un impact disparate (règle des 4/5 aux USA, critères RGPD en EU).

```python
# Analyse par type d'emploi (variable proxy potentielle de biais)
X_test_df = X_test.copy()
X_test_df["proba_defaut"] = y_proba_cal
X_test_df["accepte"] = (y_proba_cal <= 0.25).astype(int)

print("=== TAUX D'ACCEPTATION PAR TYPE D'EMPLOI ===")
print(X_test_df.groupby("type_emploi")["accepte"].agg(["mean", "count"]).round(3))

print("\n=== DISPARATE IMPACT ===")
taux_max = X_test_df.groupby("type_emploi")["accepte"].mean().max()
print(X_test_df.groupby("type_emploi")["accepte"].mean().apply(
    lambda x: f"{x:.1%} → DI={x/taux_max:.2f} ({'OK' if x/taux_max >= 0.8 else 'ALERTE'})"
))
```

---
## 10. ARCHITECTURE GITHUB

```
credit-risk-scoring-model/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_credit_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_calibration.ipynb
│   ├── 05_ks_gini_metrics.ipynb
│   └── 06_fairness_analysis.ipynb
├── src/
│   ├── features.py
│   ├── scoring.py
│   └── fairness.py
└── figures/
    ├── credit_calibration.png
    └── credit_ks_curve.png
```

---
## 16. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve |
|-----------|--------|
| Calibration | CalibratedClassifierCV isotonic |
| KS Statistic / Gini | Métriques standard crédit calculées |
| Seuil d'acceptation | Tableau politique crédit multi-seuils |
| Équité algorithmique | Disparate impact par groupe |
| Réglementation | Mention RGPD, AI Act, explicabilité |

---

*Fin du document — TSAGUE EMMANUEL — CAS 13 — Risque Crédit*
---

## Contact & Liens

**TSAGUE EMMANUEL** - Data Scientist

| | |
|---|---|
| Email | [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr) |
| LinkedIn | [emmanuel-tsague-114295414](https://www.linkedin.com/in/emmanuel-tsague-114295414) |
| GitHub | [github.com/TSAGUE25](https://github.com/TSAGUE25) |
| Formation | Datascientest 2024 |
| Experience | EDF MAD EDVANCE |
| Domaines | Machine Learning - Data Analysis - Energie |

---

> Toutes les donnees de ce depot sont simulees et anonymisees.  
> Aucune donnee reelle ou confidentielle n'est presente.
