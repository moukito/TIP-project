%% ------------------------------------------------------------
% Transfert learning avec EfficientNet-B0 sur Food-11
% ------------------------------------------------------------

clc; clear; close all;

%% ------------------------------------------------------------
% Chargement du dataset
% ------------------------------------------------------------
dataDir  = fullfile('dataset');
trainDir = fullfile(dataDir, 'train');
testDir  = fullfile(dataDir, 'test');

imds = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');
imdsTest = imageDatastore(testDir);

fprintf('Images train : %d\n', numel(imdsTrain.Files));
fprintf('Images val   : %d\n', numel(imdsVal.Files));
fprintf('Images test  : %d\n', numel(imdsTest.Files));

%% ------------------------------------------------------------
% Chargement du modèle EfficientNet-B0
% ------------------------------------------------------------
net = efficientnetb0;
inputSize  = net.Layers(1).InputSize;   % [224 224 3]
numClasses = numel(categories(imdsTrain.Labels));

% Class weights pour renforcer dairy et fried_food
classNames = categories(imdsTrain.Labels);
weights = ones(numClasses,1);
idxDairy = find(strcmp(classNames, 'dairy'));
idxFried = find(strcmp(classNames, 'fried_food'));
weights(idxDairy) = 1.1;
weights(idxFried) = 1.1;

%% ------------------------------------------------------------
% Stratégie d’amélioration ciblée
% ------------------------------------------------------------
% Dans le but d’obtenir de meilleurs résultats, nous allons adopter une approche d’amélioration ciblée plutôt que globale. L’analyse des matrices de confusion a montré que certaines classes, comme rice ou noodles-pasta, sont déjà bien reconnues malgré leur faible représentation, tandis que d’autres, notamment dairy et fried food, restent difficiles à distinguer. Plutôt que de modifier la distribution du dataset (interdiction d’après le sujet), nous concentrerons nos ajustements sur ces classes problématiques. Concrètement, nous renforcerons leur influence pendant l’entraînement en pondérant la fonction de perte (class weights) de manière à ce que les erreurs sur ces catégories comptent davantage. En parallèle, une augmentation photométrique sélective (variations légères de luminosité, de contraste et de rotation) sera appliquée uniquement aux images de ces classes, afin d’accroître la diversité visuelle perçue par le réseau tout en diminuant le temps d'entraînement. Enfin, nous avons pu réaliser des essais complémentaires grâce à la gentillesse d’un camarade qui nous a donné temporairement accès à un ordinateur plus performant. Cela nous a permis d’augmenter la résolution d’entrée de 128×128 à 224×224 pour les modèles pré-entraînés. Rien que cette modification, sans changer aucun autre hyperparamètre, a entraîné une hausse d’environ 1 % de l’accuracy sur l’ensemble de validation.

% La photométrie ne s’applique que sur : dairy, fried_food
targetClasses = ["dairy","fried_food"];

imdsTrain.ReadFcn = @(filename) readTrainSelectivePhotometry(filename, targetClasses);

% Augmentation géométrique standard (commune à toutes les classes)
imageAug = imageDataAugmenter( ...
    'RandRotation',     [-15 15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale',       [0.9 1.1], ...
    'RandYScale',       [0.9 1.1], ...
    'RandXReflection',  true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAug, ...
    'ColorPreprocessing','none');

augVal  = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing','none');
augTest = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing','none');

%% ------------------------------------------------------------
% Adaptation du modèle
% ------------------------------------------------------------
lgraph = layerGraph(net);
layers = lgraph.Layers;

idxFc = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
idxSoft = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.SoftmaxLayer'), layers), 1, 'last');
idxClass = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1, 'last');

oldFcName = layers(idxFc).Name;
oldSoftName = layers(idxSoft).Name;
oldClassName = layers(idxClass).Name;

newFcLayer = fullyConnectedLayer(numClasses, ...
    'Name','fc_food11', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newSoftmaxLayer = softmaxLayer('Name','softmax_food11');
newClassLayer   = classificationLayer('Name','classoutput', 'ClassWeights', weights);

lgraph = replaceLayer(lgraph, oldFcName, newFcLayer);
lgraph = replaceLayer(lgraph, oldSoftName, newSoftmaxLayer);
lgraph = replaceLayer(lgraph, oldClassName, newClassLayer);

%% ------------------------------------------------------------
% Options d’entraînement
% ------------------------------------------------------------
miniBatchSize = 64;
valFreq = floor(numel(imdsTrain.Files)/miniBatchSize);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',valFreq, ...
    'Verbose',true, ...
    'Plots','training-progress');

%% ------------------------------------------------------------
% Entraînement du modèle
% ------------------------------------------------------------
netEfficientNetB0 = trainNetwork(augTrain, lgraph, options);

%% ------------------------------------------------------------
% Évaluation sur validation
% ------------------------------------------------------------
[YPred,~] = classify(netEfficientNetB0, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation (EfficientNet-B0 ciblé) : %.2f%%\n', accuracy*100);

figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - EfficientNet-B0 (photométrie ciblée)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ------------------------------------------------------------
% Sauvegarde
% ------------------------------------------------------------
save('net_efficientnetb0_targeted_photometry.mat','netEfficientNetB0');
fprintf('Modèle sauvegardé : net_efficientnetb0_targeted_photometry.mat\n');

%% ------------------------------------------------------------
% ----- FONCTION LOCALE : photométrie sélective -----
% ------------------------------------------------------------
function Iout = readTrainSelectivePhotometry(filename, targetClasses)
    I = imread(filename);
    I = im2double(I);

    % Identifier la classe (nom du dossier)
    [folderPath,~] = fileparts(filename);
    [~, className] = fileparts(folderPath);

    % Appliquer la photométrie uniquement si la classe est ciblée
    if any(strcmpi(className, targetClasses))
        brightnessFactor = 0.7 + 0.6*rand(); % [0.7,1.3]
        contrastFactor   = 0.7 + 0.6*rand(); % [0.7,1.3]
        meanVal = mean(I(:));
        I = (I - meanVal) * contrastFactor + meanVal;
        I = I * brightnessFactor;
    end

    Iout = im2uint8(min(max(I,0),1));
end
