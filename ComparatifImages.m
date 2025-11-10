%% ------------------------------------------------------------
% Comparaison visuelle à même échelle : 512x512 vs 128x128 (affichées identiques)
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
% Paramètres
% ------------------------------------------------------------
inputSizeLarge = [512 512];
inputSizeSmall = [128 128];

% ------------------------------------------------------------
% Affichage côte à côte à la même échelle d'affichage
% ------------------------------------------------------------
figure('Name','Comparaison visuelle à même échelle : 512x512 vs 128x128', ...
       'NumberTitle','off', ...
       'Position',[100 100 1200 800]);

for i = 1:numImages
    I = imread(sampleFiles{i});
    
    % Redimensionnement
    I_large = imresize(I, inputSizeLarge);   % référence 512x512
    I_small = imresize(I, inputSizeSmall);   % réduction à 128x128
    I_small_upscaled = imresize(I_small, inputSizeLarge, 'bicubic'); % ré-agrandi à 512x512 pour affichage
    
    % Affichage côte à côte à même taille d'affichage
    subplot(3,4,i);
    imshowpair(I_large, I_small_upscaled, 'montage');
    title(sprintf('%s\n512x512 (gauche)  vs  128x128 upscalé (droite)', string(sampleLabels(i))), ...
        'FontSize', 9, 'Interpreter', 'none');
end

sgtitle('Comparaison à échelle identique : images originales 512x512 vs versions 128x128 (Food-11)', 'FontSize', 14);

fprintf('Affichage terminé : 12 comparaisons côte à côte (512 vs 128, même échelle visuelle).\n');
