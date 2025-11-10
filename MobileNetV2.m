%% ------------------------------------------------------------
% Transfert learning avec MobileNetV2 sur Food-11
% ------------------------------------------------------------

clc; clear; close all;

%% ------------------------------------------------------------
% Chargement du dataset
% ------------------------------------------------------------
dataDir  = fullfile('dataset');
trainDir = fullfile(dataDir, 'train');
testDir  = fullfile(dataDir, 'test');

% Datastore (train + labels)
imds = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split 90% / 10% pour train et validation
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');

% Test set (sans labels)
imdsTest = imageDatastore(testDir);

fprintf('Images train : %d\n', numel(imdsTrain.Files));
fprintf('Images val   : %d\n', numel(imdsVal.Files));
fprintf('Images test  : %d\n', numel(imdsTest.Files));

%% ------------------------------------------------------------
% Statistiques sur le dataset (répartition des classes)
% ------------------------------------------------------------
tblTrain = countEachLabel(imdsTrain);
tblVal   = countEachLabel(imdsVal);

disp('Répartition des classes - Train :');
disp(tblTrain);

disp('Répartition des classes - Validation :');
disp(tblVal);

fprintf('Train - nb moyen d''images/classe : %.1f (min = %d, max = %d)\n', ...
    mean(tblTrain.Count), min(tblTrain.Count), max(tblTrain.Count));
fprintf('Val   - nb moyen d''images/classe : %.1f (min = %d, max = %d)\n', ...
    mean(tblVal.Count), min(tblVal.Count), max(tblVal.Count));

%% ------------------------------------------------------------
% Chargement du modèle pré-entraîné MobileNetV2
% ------------------------------------------------------------
net = mobilenetv2;   % nécessite Deep Learning Toolbox Model for MobileNetv2

inputSize  = net.Layers(1).InputSize;   % normalement [224 224 3]
numClasses = numel(categories(imdsTrain.Labels));

%% ------------------------------------------------------------
% Data augmentation
%   - Photométrique (lumière/contraste) via ReadFcn sur imdsTrain
%   - Géométrique via imageDataAugmenter
% ------------------------------------------------------------

% 1) Photométrique : on modifie la fonction de lecture du TRAIN uniquement
imdsTrain.ReadFcn = @readTrainWithColorAug;

% 2) Augmentation géométrique (rotation, translation, zoom, flip)
imageAug = imageDataAugmenter( ...
    'RandRotation',     [-15 15], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale',       [0.9 1.1], ...
    'RandYScale',       [0.9 1.1], ...
    'RandXReflection',  true);

% Train : couleur (ReadFcn) + géométrie + resize à 224x224
augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAug, ...
    'ColorPreprocessing','none');

% Val / Test : pas d'augmentation couleur, juste resize
augVal  = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing','none');

augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing','none');

%% ------------------------------------------------------------
% Adaptation de MobileNetV2 à 11 classes (Food-11)
% ------------------------------------------------------------
lgraph = layerGraph(net);

% Nouvelles couches de sortie
newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name','fc_food11', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax_food11')
    classificationLayer('Name','classoutput')
];

% Pour MobileNetV2, les 3 dernières couches s'appellent :
% 'Logits' -> 'Logits_softmax' -> 'ClassificationLayer_Logits'
lgraph = replaceLayer(lgraph, 'Logits',                  newLayers(1));
lgraph = replaceLayer(lgraph, 'Logits_softmax',          newLayers(2));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newLayers(3));

%% ------------------------------------------------------------
% Options d'entraînement (transfert learning)
% ------------------------------------------------------------
miniBatchSize = 64;
valFreq = floor(numel(imdsTrain.Files) / miniBatchSize);

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...   % bien adapté pour du fine-tuning
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', valFreq, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% ------------------------------------------------------------
% Entraînement du modèle MobileNetV2 adapté
% ------------------------------------------------------------
netMobileNetV2 = trainNetwork(augTrain, lgraph, options);

%% ------------------------------------------------------------
% Évaluation sur la validation
% ------------------------------------------------------------
[YPred, ~] = classify(netMobileNetV2, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation (MobileNetV2, transfert learning) : %.2f%%\n', accuracy * 100);

figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - MobileNetV2 (transfert learning Food-11)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ------------------------------------------------------------
% Sauvegarde du réseau entraîné
% ------------------------------------------------------------
save('net_mobilenetv2_food11.mat', 'netMobileNetV2');
fprintf('Modèle sauvegardé sous : net_mobilenetv2_food11.mat\n');

%% ------------------------------------------------------------
% ----- FONCTION LOCALE : lecture + augmentation couleur -----
% ------------------------------------------------------------
function Iout = readTrainWithColorAug(filename)
    I = imread(filename);
    I = im2double(I);

    % Variation luminosité
    brightnessFactor = 0.8 + 0.4*rand();    % [0.8, 1.2]
    I = I * brightnessFactor;

    % Variation contraste
    contrastFactor = 0.8 + 0.4*rand();      % [0.8, 1.2]
    meanVal = mean(I(:));
    I = (I - meanVal) * contrastFactor + meanVal;

    % Clamp + retour en uint8
    Iout = im2uint8(min(max(I,0),1));
end
