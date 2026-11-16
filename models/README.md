# Modèles entraînés

Les deux modèles fonctionnent sans dataset ni nouvel entraînement.
Dans Streamlit, choisir **svm** et renseigner le chemin du modèle.

| Modèle | Reçus d’entraînement | Utilisation |
| --- | --- | --- |
| `receipt-svm.joblib` | 521 SROIE | Modèle par défaut |
| `receipt-svm-general.joblib` | 521 SROIE + 1 077 WildReceipt | Essayer d’autres formats |

Les reçus de validation et de test sont séparés de l’entraînement.
Les deux modèles utilisent Tesseract et TF-IDF + SVM. Le second ajoute la position
du texte, les lignes voisines, les blocs multilignes et la calibration.

## Résultats sur les jeux de test

Le score ci-dessous est la moyenne des F1 des quatre champs, pas un taux de reçus
entièrement corrects.

| Jeu de test | Modèle initial | Modèle général |
| --- | --- | --- |
| SROIE : 344 reçus | 51,76 % | 48,32 % |
| WildReceipt : 472 reçus | 25,25 % | 34,12 % |

Le modèle général améliore WildReceipt mais réduit le score SROIE.
Les champs `company` et `address` restent les plus difficiles.

## Détails et compatibilité

Utiliser `uv sync --locked` pour conserver les versions compatibles.
Les scores SVM du modèle initial ne sont pas des probabilités. La calibration du
modèle général ne garantit pas qu’un champ extrait soit correct.

- [Comparaison des étapes et commandes](../docs/svm-improvements.md).
- [Rapport du modèle initial](receipt-svm.report.json).
- [Rapport du modèle général](receipt-svm-general.report.json).
- [Scores détaillés par champ et par étape](svm-stages.report.json).
