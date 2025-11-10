%% ------------------------------------------------------------
% CNN - LeNet-5 
% ------------------------------------------------------------

clc; clear; close all;

% ------------------------------------------------------------
% Chargement du dataset
% ------------------------------------------------------------
dataDir = fullfile('dataset');
trainDir = fullfile(dataDir, 'train');
testDir  = fullfile(dataDir, 'test');

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

% Redimensionnement automatique pour garantir la même taille
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

% ------------------------------------------------------------
% Architecture du LeNet-5
% ------------------------------------------------------------
layers = [
    imageInputLayer(inputSize, 'Normalization','rescale-zero-one', 'Name','input')

    % 'Normalization','rescale-zero-one' normalise les pixels entre 0 et 1
    
    % Bloc 1
    convolution2dLayer(5, 6, 'Padding','same', 'Name','conv1')
    reluLayer('Name','relu1')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool1')
    
    % Bloc 2
    convolution2dLayer(5, 16, 'Padding','same', 'Name','conv2')
    reluLayer('Name','relu2')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool2')
    
    % Bloc 3 (optionnel mais utile pour images grandes)
    convolution2dLayer(5, 120, 'Padding','same', 'Name','conv3')
    reluLayer('Name','relu3')

    % Couches fully connected
    fullyConnectedLayer(84, 'Name','fc1')
    reluLayer('Name','relu_fc1')

    fullyConnectedLayer(numClasses, 'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

% ------------------------------------------------------------
% Options d'entraînement
% ------------------------------------------------------------
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...  % léger compromis entre stabilité et mémoire
    'InitialLearnRate', 5e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% ------------------------------------------------------------
% Entraînement du modèle
% ------------------------------------------------------------
netLeNet = trainNetwork(augTrain, layers, options);

% ------------------------------------------------------------
% Évaluation sur la validation
% ------------------------------------------------------------
[YPred, ~] = classify(netLeNet, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation : %.2f%%\n', accuracy * 100);

% Matrice de confusion
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - LeNet-5 (512x512)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% ------------------------------------------------------------
% Prédictions sur le test set + génération JSON
% ------------------------------------------------------------
[YPredTest, ~] = classify(netLeNet, augTest);

fprintf('Génération du fichier JSON...\n');
filePaths = imdsTest.Files;
n = numel(filePaths);
jsonStruct = struct;

for i = 1:n
    [~, name, ~] = fileparts(filePaths{i});
    jsonStruct.(name) = char(YPredTest(i));
end

jsonText = jsonencode(jsonStruct);

fid = fopen('predictions_lenet5.json','w');
fwrite(fid, jsonText, 'char');
fclose(fid);

fprintf('Fichier predictions_lenet5.json généré avec succès !\n');
