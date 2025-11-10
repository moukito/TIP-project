%% ------------------------------------------------------------
% CNN de base – Réseau maison (SimpleCNN)
% ------------------------------------------------------------

clc; clear; close all;

% ------------------------------------------------------------
% Chargement du dataset
% ------------------------------------------------------------
dataDir = fullfile('dataset');
trainDir = fullfile(dataDir, 'train');
testDir  = fullfile(dataDir, 'test');
f = figure;
r = rendererinfo(f)

% Création du datastore (train + labels)
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

% ------------------------------------------------------------
% Paramètres du modèle
% ------------------------------------------------------------
inputSize = [512 512 3];
numClasses = numel(categories(imdsTrain.Labels));

% Redimensionnement automatique pour garantir la taille identique
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% ------------------------------------------------------------
% Architecture du CNN
% ------------------------------------------------------------
layers = [
    imageInputLayer(inputSize, 'Name','input')

    convolution2dLayer(3, 16, 'Padding','same', 'Name','conv1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool1')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool2')

    convolution2dLayer(3, 64, 'Padding','same', 'Name','conv3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2, 'Stride',2, 'Name','pool3')

    fullyConnectedLayer(256, 'Name','fc1')
    reluLayer('Name','relu_fc1')

    fullyConnectedLayer(numClasses, 'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')
];

% ------------------------------------------------------------
% Options d'entraînement
% ------------------------------------------------------------
options = trainingOptions('adam', ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 4, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% ------------------------------------------------------------
% Entraînement du modèle
% ------------------------------------------------------------
netSimple = trainNetwork(augTrain, layers, options);

% ------------------------------------------------------------
% Évaluation sur la validation
% ------------------------------------------------------------
[YPred, ~] = classify(netSimple, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation : %.2f%%\n', accuracy * 100);

% Matrice de confusion
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - SimpleCNN (512x512, sans dropout)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% ------------------------------------------------------------
% Prédictions sur le test set
% ------------------------------------------------------------
[YPredTest, ~] = classify(netSimple, augTest);

% Génération du JSON pour la soumission
fprintf('Génération du fichier JSON...\n');

filePaths = imdsTest.Files;
n = numel(filePaths);
jsonStruct = struct;

for i = 1:n
    [~, name, ~] = fileparts(filePaths{i});
    jsonStruct.(name) = char(YPredTest(i));
end

jsonText = jsonencode(jsonStruct);

fid = fopen('predictions.json','w');
fwrite(fid, jsonText, 'char');
fclose(fid);

fprintf('Fichier predictions.json généré avec succès !\n');
