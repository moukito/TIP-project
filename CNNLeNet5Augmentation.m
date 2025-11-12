%% ------------------------------------------------------------
% CNN - LeNet-5 (128x128, augmentation géométrique + lumière)
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
inputSize  = [128 128 3];      % images 128x128
numClasses = numel(categories(imdsTrain.Labels));

%% ------------------------------------------------------------
% Data augmentation
%   - Photométrique (lumière/couleur) via ReadFcn sur imdsTrain
%   - Géométrique via imageDataAugmenter
% ------------------------------------------------------------

% Photométrique : on modifie la fonction de lecture du TRAIN
imdsTrain.ReadFcn = @readTrainWithColorAug;
% Validation / test : lecture standard (ReadFcn par défaut)

% Augmentation géométrique adaptée à 128x128
geomAug = imageDataAugmenter( ...
    'RandRotation',     [-15 15], ...
    'RandXTranslation', [-10 10], ...   % quelques pixels sur 128
    'RandYTranslation', [-10 10], ...
    'RandXScale',       [0.9 1.1], ...
    'RandYScale',       [0.9 1.1], ...
    'RandXReflection',  true);

% Train : couleur (ReadFcn) + géométrie + resize 128x128
augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', geomAug);

% Val / Test : simplement redimensionnés en 128x128, sans augmentation
augVal  = augmentedImageDatastore(inputSize, imdsVal);
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% ------------------------------------------------------------
% Visualisation de quelques images augmentées
% ------------------------------------------------------------
figure('Name','Exemples d''images augmentées (128x128, géométrie + lumière)');
for i = 1:6
    dataBatch = read(augTrain);
    I = dataBatch.input{1};   % on prend le contenu de la cellule
    subplot(2,3,i);
    imshow(I);
    title('Augmentée 128x128');
end
reset(augTrain);

%% ------------------------------------------------------------
% Architecture du LeNet-5 adapté à 128x128
% ------------------------------------------------------------
% 128x128 -> pool1 -> 64x64 -> pool2 -> 32x32
layers = [
    imageInputLayer(inputSize, ...
        'Normalization','rescale-zero-one', ...
        'Name','input')

    convolution2dLayer(5, 6, 'Padding','same', 'Name','conv1')
    reluLayer('Name','relu1')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool1')   % 128 -> 64

    convolution2dLayer(5, 16, 'Padding','same', 'Name','conv2')
    reluLayer('Name','relu2')
    averagePooling2dLayer(2, 'Stride',2, 'Name','avgpool2')   % 64 -> 32

    convolution2dLayer(5, 120, 'Padding','same', 'Name','conv3')
    reluLayer('Name','relu3')
    % features ~ 32x32x120

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
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...          % images moyennes, batch raisonnable
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% ------------------------------------------------------------
% Entraînement du modèle
% ------------------------------------------------------------
netLeNet_128_augColor = trainNetwork(augTrain, layers, options);

%% ------------------------------------------------------------
% Évaluation sur la validation
% ------------------------------------------------------------
[YPred, ~] = classify(netLeNet_128_augColor, augVal);
YTrue = imdsVal.Labels;

accuracy = mean(YPred == YTrue);
fprintf('Accuracy validation (128x128, géométrie + lumière) : %.2f%%\n', accuracy * 100);

% Matrice de confusion
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - LeNet-5 (128x128 + augmentation géométrique + lumière)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ------------------------------------------------------------
% Sauvegarde du réseau entraîné
% ------------------------------------------------------------
save('net_lenet5_128_aug_color.mat', 'netLeNet_128_augColor');
fprintf('Modèle sauvegardé sous : net_lenet5_128_aug_color.mat\n');

%% ------------------------------------------------------------
% ----- FONCTION LOCALE : lecture + augmentation couleur -----
% ------------------------------------------------------------
function Iout = readTrainWithColorAug(filename)
    % Lecture de l'image
    I = imread(filename);

    % Conversion en double [0,1]
    I = im2double(I);

    % Variation aléatoire de luminosité (multiplicatif)
    brightnessFactor = 0.8 + 0.4*rand();      % [0.8, 1.2]
    I = I * brightnessFactor;

    % Variation de contraste autour de la moyenne
    contrastFactor = 0.8 + 0.4*rand();        % [0.8, 1.2]
    meanVal = mean(I(:));
    I = (I - meanVal) * contrastFactor + meanVal;

    % Variation de saturation via HSV
    if size(I,3) == 3
        hsv = rgb2hsv(I);
        satFactor = 0.8 + 0.4*rand();         % [0.8, 1.2]
        hsv(:,:,2) = max(0, min(1, hsv(:,:,2) * satFactor));
        I = hsv2rgb(hsv);
    end

    % Clamp + retour en uint8 (compatible toolbox)
    Iout = im2uint8(min(max(I,0),1));
end
