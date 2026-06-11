import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                              roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

NUM_FEATURES = ['age', 'revenu_annuel', 'montant_credit', 'duree_mois',
                'anciennete_emploi', 'nb_credits_en_cours', 'taux_endettement']
CAT_FEATURES = ['historique', 'type_credit']
HIST_CATS   = [['bon', 'neutre', 'mauvais']]
TYPE_CATS   = [['immobilier', 'auto', 'personnel', 'revolving']]


def encode_categoricals(df):
    df = df.copy()
    for col in ['historique', 'type_credit']:
        df[col] = df[col].astype('category').cat.codes
    return df


def build_preprocessor():
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CAT_FEATURES),
    ])


def build_pipeline(model_name='xgb'):
    pre = build_preprocessor()
    if model_name == 'xgb' and XGB_AVAILABLE:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              scale_pos_weight=4, random_state=42, verbosity=0,
                              eval_metric='logloss')
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight='balanced', random_state=42)
    elif model_name == 'gbm':
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.1, random_state=42)
    else:
        model = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
    return Pipeline([('preprocessor', pre), ('model', model)])


def compare_models(X_train, y_train, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    for name in ['logistic', 'rf', 'gbm']:
        scores = cross_val_score(build_pipeline(name), X_train, y_train,
                                 cv=skf, scoring='roc_auc')
        results[name] = {'AUC_mean': scores.mean(), 'AUC_std': scores.std()}
    if XGB_AVAILABLE:
        scores = cross_val_score(build_pipeline('xgb'), X_train, y_train,
                                 cv=skf, scoring='roc_auc')
        results['xgb'] = {'AUC_mean': scores.mean(), 'AUC_std': scores.std()}
    df = pd.DataFrame(results).T.sort_values('AUC_mean', ascending=False)
    print(df.round(4))
    return df


def evaluate_credit_model(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    print(f'ROC-AUC = {auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Solvable', 'Défaut']))
    _plot_roc_ks(y_test, y_proba, auc)
    return dict(auc=auc, y_proba=y_proba)


def gini_coefficient(y_true, y_proba):
    return 2 * roc_auc_score(y_true, y_proba) - 1


def ks_statistic(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return np.max(tpr - fpr)


def _plot_roc_ks(y_test, y_proba, auc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC={auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.set_title('Courbe ROC'); ax1.legend()

    sorted_idx = np.argsort(y_proba)[::-1]
    cum_pos = np.cumsum(y_test.values[sorted_idx]) / y_test.sum()
    cum_neg = np.cumsum(1 - y_test.values[sorted_idx]) / (1 - y_test).sum()
    ks = np.max(cum_pos - cum_neg)
    pct = np.linspace(0, 1, len(y_proba))
    ax2.plot(pct, cum_pos, color='#F44336', label='Défauts cumulés')
    ax2.plot(pct, cum_neg, color='#4CAF50', label='Solvables cumulés')
    ax2.set_title(f'Courbe KS (KS={ks:.3f})'); ax2.legend()
    ax2.set_xlabel('% population triée par score')

    plt.suptitle('Évaluation Credit Scoring', fontweight='bold')
    plt.tight_layout(); plt.show()


def plot_score_distribution(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.6, color='#4CAF50', label='Solvable')
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.7, color='#F44336', label='Défaut')
    ax.set_xlabel('Score de risque (probabilité de défaut)')
    ax.set_title('Distribution des scores par classe', fontweight='bold')
    ax.legend(); plt.tight_layout(); plt.show()
