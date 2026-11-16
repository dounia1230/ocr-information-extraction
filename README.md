# OCR Information Extraction

Extrait quatre champs d’un reçu image ou PDF : **company, date, address, total**.

**Image/PDF → Tesseract → Regex ou SVM → JSON**, en local sur CPU.
Formats acceptés : PNG, JPG, JPEG et PDF scanné. Un champ incertain vaut `null`.

## Installation

Installer Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
et Tesseract avec la langue anglaise. Sous Windows, ajouter Tesseract au PATH.

```sh
git clone https://github.com/dounia1230/ocr-information-extraction.git
cd ocr-information-extraction
uv sync --locked
```

## Lancer l’interface

```sh
uv run streamlit run app.py
```

Ouvrir l’URL affichée, choisir **svm**, importer un reçu puis cliquer sur
**Extract information**. Deux modèles entraînés sont fournis :

| Chemin dans « Local trained model » | Entraînement |
| --- | --- |
| `models/receipt-svm.joblib` (par défaut) | SROIE |
| `models/receipt-svm-general.joblib` | SROIE + WildReceipt |

Aucun entraînement ni téléchargement de dataset n’est nécessaire pour les essayer.
Le premier est meilleur sur SROIE ; le second est meilleur sur WildReceipt.

## Utiliser la ligne de commande

```sh
uv run ocr-extract extract receipt.jpg --method svm --model models/receipt-svm-general.joblib --output result.json
```

Sans `--method svm`, l’extraction utilise Regex. Prétraitements facultatifs :
`--grayscale`, `--denoise`, `--threshold`, `--deskew`.

## Entraînement, résultats et tests

- [Entraîner et comparer les modèles](docs/svm-improvements.md).
- [Résumé des modèles et scores](models/README.md).

```sh
uv run pytest -q
```

Les noms d’entreprise et adresses restent difficiles sur certains reçus.
Les dates ambiguës sont interprétées jour/mois. Charger uniquement des modèles
`.joblib` de confiance.
