# TIP-project

## Présentation

Ce projet implémente une solution d'apprentissage automatique pour la classification d'images de nourriture utilisant le dataset Food-11. L'objectif de ce projet est de produire un fichier JSON de prédictions pour évaluation sur un ensemble de test, pour pouvoir le comparer avec le serveur test.

## Dataset

- **Images totales** : 16 393
- **Catégories** : 11 (Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit)
- **Ensemble d'entraînement** : 13 296 images (organisées en répertoires de classes)
- **Ensemble de test** : 3 097 images (sans étiquettes de classe)

## Approche

Le projet explore différentes architectures CNN : modèles from scratch (LeNet-5), puis transfert learning avec EfficientNet-B0, ResNet-50 et MobileNetV2. Différentes techniques ont été utilisées :

- Augmentation de données (photométrique et géométrique)
- Pondération des classes dans la fonction de Loss pour équilibrer les catégories difficiles

## Fichiers du dépôt

### Scripts MATLAB (.m)
- `Final.m` : Stratégie d'amélioration ciblée avec EfficientNet-B0, incluant pondération des classes et augmentation sélective
- `PreTrained-Resnet50.m` : Transfert learning avec ResNet-50
- `PreTrained-MobileNetV2.m` : Transfert learning avec MobileNetV2
- `PreTrained-EfficientNetB0.m` : Transfert learning avec EfficientNet-B0
- `LeNet-5.m` : Implémentation personnalisée de LeNet-5 from scratch
- `LeNet-5_Augmentation.m` : LeNet-5 avec augmentation de données
- `ComparatifImages.m` : Script de comparaison d'images
- `JsonCreator.m` : Générateur de fichier de prédictions JSON

### Fichiers de données
- `predictions.json` : Prédictions de base du modèle
- `predictions_combined.json` : Prédictions combinées
- `test_prediction.json` : Prédictions finales perturbées
- Fichiers de modèles sauvegardés (.mat)

## Installation et exécution

### Prérequis

- MATLAB (pour les scripts .m) avec Deep Learning Toolbox

### Étapes

1. Placez le dataset dans le fichier racine.
2. Exécutez un script MATLAB comme `PreTrained-Resnet18.m` pour entraîner le modèle.
3. Exécutez ensuite `JsonCreator.m`  pour generer les Jsons.