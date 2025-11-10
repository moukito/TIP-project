%% ------------------------------------------------------------
% CNN - LeNet-5 (224x224 avec data augmentation avancée)
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
% Paramètres du modèle
% ------------------------------------------------------------
inputSize  = [224 224 3];
numClasses = numel(categories(imdsTrain.Labels));

%% ------------------------------------------------------------
% Data augmentation (géométrique + photométrique)
% ------------------------------------------------------------
imageAug = imageDataAugmenter( ...
    'RandRotation',       [-15 15], ...
    'RandXTranslation',   [-20 20], ...
    'RandYTranslation',   [-20 20], ...
    'RandXScale',         [0.9 1.1], ...
    'RandYScale',         [0.9 1.1], ...
    'RandXReflection',    true, ...
    'RandYReflection',    false, ...
    'RandBrightness',     [0.8 1.2], ...     % variations de luminosité
    'RandContrast',       [0.8 1.2], ...     % variations de contraste
    'RandSaturation',     [0.7 1.3], ...     % variations de saturation
    'RandHue',            [-0.05 0.05]);     % petites variations de teinte

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAug);

augVal  = augmentedImageDatastore(inputSize, imdsVal);
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% ------------------------------------------------------------
% Visualisation de quelques images augmentées
% ------------------------------------------------------------
figure('Name','Exemples d''images augmentées');
for i = 1:6
    I = read(augTrain);
    subplot(2,3,i);
    imshow(I.input);
    title('Exemple augmenté');
end
reset(augTrain);

%% ------------------------------------------------------------
% Architecture du LeNet-5 adapté à 224x224
% ------------------------------------------------------------
layers = [
    imageInputLayer(inputSize, ...
        'Normalization','rescale-zero-one', ...
        'Name','input')

    convolution2dLayer(5, 6, 'Padding','same', 'Name','conv1')
    reluLayer('Name','relu1')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool1')

    convolution2dLayer(5, 16, 'Padding','same', 'Name','conv2')
    reluLayer('Name','relu2')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool2')

    convolution2dLayer(5, 120, 'Padding','same', 'Name','conv3')
    reluLayer('Name','relu3')

    fullyConnectedLayer(84, 'Name','fc1')
    reluLayer('Name','relu_fc1')

    fullyConnectedLayer(numClasses, 'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

%% ------------------------------------------------------------
% Options d'entraînement
% ------------------------------------------------------------
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% ------------------------------------------------------------
% Entraînement du modèle
% ------------------------------------------------------------
netLeNet_224_augColor = trainNetwork(augTrain, layers, options);

%% ------------------------------------------------------------
% Évaluation sur la validation
% ------------------------------------------------------------
[YPred, ~] = classify(netLeNet_224_augColor, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation (224x224, aug + couleur) : %.2f%%\n', accuracy * 100);

% Matrice de confusion
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - LeNet-5 (224x224 + augmentation couleur)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ------------------------------------------------------------
% Sauvegarde du réseau entraîné
% ------------------------------------------------------------
save('net_lenet5_224_aug_color.mat', 'netLeNet_224_augColor');
fprintf('Modèle sauvegardé sous : net_lenet5_224_aug_color.mat\n');
