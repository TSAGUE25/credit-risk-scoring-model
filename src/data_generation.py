import numpy as np
import pandas as pd
from pathlib import Path


def generate_credit_data(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 70, n)
    revenu_annuel = np.clip(rng.lognormal(10.5, 0.5, n), 15000, 300000).round(-2)
    montant_credit = np.clip(rng.lognormal(10.2, 0.7, n), 1000, 250000).round(-2)
    duree_mois = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n,
                            p=[0.05, 0.10, 0.20, 0.15, 0.20, 0.10, 0.10, 0.05, 0.05])
    anciennete_emploi = np.clip(rng.exponential(5, n), 0, 40).round(1)
    nb_credits_en_cours = rng.integers(0, 6, n)
    taux_endettement = np.clip((montant_credit / duree_mois * 12) / revenu_annuel * 100,
                               0, 80).round(1)
    historique = rng.choice(['bon', 'neutre', 'mauvais'], n, p=[0.55, 0.30, 0.15])
    type_credit = rng.choice(['personnel', 'immobilier', 'auto', 'revolving'], n,
                             p=[0.35, 0.30, 0.20, 0.15])

    # Default probability model
    hist_score = np.where(historique == 'bon', -0.5,
                 np.where(historique == 'neutre', 0.2, 1.0))
    logit = (
        -3.0
        + hist_score
        + 0.03 * taux_endettement
        + 0.15 * nb_credits_en_cours
        - 0.02 * anciennete_emploi
        + np.where(type_credit == 'revolving', 0.5, 0)
        + rng.normal(0, 0.3, n)
    )
    prob_defaut = 1 / (1 + np.exp(-logit))
    defaut = (rng.uniform(0, 1, n) < prob_defaut).astype(int)

    return pd.DataFrame({
        'age': age,
        'revenu_annuel': revenu_annuel.astype(int),
        'montant_credit': montant_credit.astype(int),
        'duree_mois': duree_mois,
        'anciennete_emploi': anciennete_emploi,
        'nb_credits_en_cours': nb_credits_en_cours,
        'taux_endettement': taux_endettement,
        'historique': historique,
        'type_credit': type_credit,
        'defaut': defaut,
    })


def load_or_generate(csv_path, n=10000, seed=42):
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    df = generate_credit_data(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
