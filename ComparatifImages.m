%% ------------------------------------------------------------
% Comparaison côte à côte : 512x512 vs 224x224
% ------------------------------------------------------------

clc; clear; close all;

% ------------------------------------------------------------
% Chargement du dataset
% ------------------------------------------------------------
dataDir = fullfile('dataset', 'train');

% Datastore pour récupérer des images aléatoires
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Sélection de 12 images aléatoires
numImages = 12;
idx = randperm(numel(imds.Files), numImages);
sampleFiles = imds.Files(idx);
sampleLabels = imds.Labels(idx);

% ------------------------------------------------------------
% Paramètres de redimensionnement
% ------------------------------------------------------------
inputSizeLarge = [512 512];
inputSizeSmall = [224 224];

% ------------------------------------------------------------
% Affichage côte à côte
% ------------------------------------------------------------
figure('Name','Comparaison visuelle 512x512 vs 224x224', ...
       'NumberTitle','off', ...
       'Position',[100 100 1200 800]);

for i = 1:numImages
    I = imread(sampleFiles{i});
    I_large = imresize(I, inputSizeLarge);
    I_small = imresize(I, inputSizeSmall);

    % Affichage côte à côte
    subplot(3,4,i);
    imshowpair(I_large, I_small, 'montage');
    title(sprintf('%s\n512x512  ←→  224x224', string(sampleLabels(i))), ...
          'FontSize', 10, 'Interpreter', 'none');
end

sgtitle('Comparaison d''échelle : images 512x512 vs 224x224 (Food-11)', 'FontSize', 14);

fprintf('Affichage terminé : 12 comparaisons côte à côte (512 vs 224).\n');
