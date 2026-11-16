# Entraînement et comparaison du SVM

## Améliorations testées

1. Utiliser les annotations et leurs positions pour mieux étiqueter les lignes OCR.
2. Ajouter la position du texte et les lignes voisines aux features TF-IDF.
3. Regrouper les noms et adresses sur plusieurs lignes.
4. Ajouter [WildReceipt](https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/wildreceipt/metafile.yml) à SROIE.
5. Calibrer les scores et choisir les seuils sur la validation.

Tesseract reste sur CPU, en anglais, PSM 6. Les reçus de démonstration ne servent
pas à l’entraînement. Aucun nom de commerce ni montant de reçu n’est codé en dur.

## Données

| Dataset | Entraînement | Validation | Test |
| --- | --- | --- | --- |
| SROIE | 521 | 92 | 344 |
| WildReceipt | 1 077 | 190 | 472 |

Séparation par reçu, graine 42. Les doublons d’image identiques sont exclus ; les
datasets et caches OCR restent locaux. Les images recadrées ou redimensionnées
peuvent échapper à cette vérification.

## Résultats

F1 moyen des champs `company`, `date`, `address`, `total` sur les jeux de test :

| Étape | SROIE | WildReceipt |
| --- | --- | --- |
| Regex | 40,52 % | 31,82 % |
| SVM initial | 51,76 % | 25,25 % |
| 1. Meilleures étiquettes | 49,15 % | 29,63 % |
| 2. Position et contexte | 49,36 % | 29,08 % |
| 3. Blocs multilignes | 48,77 % | 28,92 % |
| 4. Deux datasets | 48,71 % | 35,76 % |
| 5. Calibration | 48,32 % | 34,12 % |

Le modèle initial reste celui par défaut : il gagne sur la validation SROIE.
Le modèle général correspond à l’étape 5, choisie sur la validation combinée avant
les tests. L’étape 4 obtient ensuite un meilleur score de test ; chaque étape
n’apporte donc pas forcément un gain.

Les [rapports JSON](../models/svm-stages.report.json) conservent précision, rappel,
F1, correspondance exacte et temps par champ/étape. Une valeur incorrecte compte
comme faux positif et, si une valeur était attendue, comme faux négatif.
Sur WildReceipt, les espaces sont ignorés pour les noms/adresses et les dates
numériques ambiguës sont exclues du score. Les annotations sans valeur numérique
ou date unique interprétable sont également exclues. Comparer les modèles au sein
d’un même dataset.

## Réentraîner (facultatif)

Télécharger et extraire [SROIE](https://www.kaggle.com/datasets/tatevikhakobyan/sroie-datasetv1)
dans `data/sroie/SROIE2019` et l’[archive officielle WildReceipt](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)
dans `data/public/wildreceipt`.

Exécuter chaque commande sur une ligne, dans cet ordre :

```sh
uv run ocr-extract prepare data/sroie/SROIE2019 --output data/processed --seed 42 --deduplicate --exclude-document train/img/X51005663280.jpg
uv run ocr-extract prepare-wildreceipt data/public/wildreceipt --output data/wildreceipt-processed --workers 4 --seed 42
uv run ocr-extract merge data/processed/prepared.json data/wildreceipt-processed/prepared.json --output data/combined/prepared.json
uv run ocr-extract train data/combined/prepared.json --stage calibrated --c 0.5 --model models/my-svm.joblib
uv run ocr-extract compare data/combined/prepared.json --models models/my-svm.joblib --split validation --output outputs/validation.json
uv run ocr-extract compare data/combined/prepared.json --models models/my-svm.joblib --split test --output outputs/test.json
```

`--exclude-document` écarte un reçu dont l’annotation d’adresse manque.
Attendre la fin de la préparation avant l’entraînement. Si le cache SROIE existe
déjà, sauter sa préparation. Pour un ancien cache sans positions de référence,
l’enrichir avant la fusion :

```sh
uv run ocr-extract enrich data/processed/prepared.json --dataset data/sroie/SROIE2019 --output data/processed/spatial.json
```

Dans ce cas, utiliser `data/processed/spatial.json` comme premier fichier de
`merge`. Choisir les réglages sur la validation, puis consulter le test final.
Pour reproduire les étapes intermédiaires, `train --stage` accepte `references`,
`spatial`, `blocks` et `calibrated` ; l’étape 4 utilise `blocks` avec les deux datasets.

Les nouveaux modèles sont enregistrés sans remplacer ceux fournis. Pour refaire
la préparation, choisir un autre dossier de sortie. Les images difficiles et les
erreurs OCR peuvent toujours produire des champs incorrects ou `null`.
