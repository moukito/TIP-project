% ------------------------------------------------------------
% JsonCreatorBatch.m
% Génère automatiquement un JSON pour chaque modèle .mat du dossier
% ------------------------------------------------------------

clc; clear; close all;

%% ------------------------------------------------------------
% Trouver tous les fichiers .mat du dossier courant
% ------------------------------------------------------------
matFiles = dir("*.mat");

if isempty(matFiles)
    error("Aucun fichier .mat trouvé dans ce dossier.");
end

fprintf("Fichiers détectés :\n");
for k = 1:numel(matFiles)
    fprintf(" - %s\n", matFiles(k).name);
end
fprintf("\n");

%% ------------------------------------------------------------
% Boucle sur chaque fichier modèle
% ------------------------------------------------------------
for k = 1:numel(matFiles)

    matFile = matFiles(k).name;
    fprintf("\n=========================================\n");
    fprintf("Traitement du modèle : %s\n", matFile);
    fprintf("=========================================\n");

    % Charger le .mat
    S = load(matFile);

    % Détecter automatiquement la variable contenant le réseau
    names = fieldnames(S);
    net = [];
    
    for i = 1:numel(names)
        obj = S.(names{i});
        if isa(obj, "SeriesNetwork") || isa(obj, "DAGNetwork") || isa(obj, "dlnetwork")
            net = obj;
            fprintf("Réseau détecté : %s\n", names{i});
            break;
        end
    end
    
    if isempty(net)
        fprintf("Aucun réseau valide trouvé dans %s — ignoré.\n", matFile);
        continue;
    end

    %% ------------------------------------------------------------
    % Recharger le test set
    % ------------------------------------------------------------
    dataDir = fullfile("dataset");
    testDir = fullfile(dataDir, "test");
    
    imdsTest = imageDatastore(testDir);

    inputSize = net.Layers(1).InputSize;
    
    augTest = augmentedImageDatastore(inputSize, imdsTest, ...
        "ColorPreprocessing","none");

    %% ------------------------------------------------------------
    % Classification
    % ------------------------------------------------------------
    fprintf("Classification du test...\n");
    YPredTest = classify(net, augTest);

    %% ------------------------------------------------------------
    % Génération du JSON
    % ------------------------------------------------------------
    filePaths = imdsTest.Files;
    n = numel(filePaths);

    jsonMap = containers.Map('KeyType','char', 'ValueType','char');

    for i = 1:n
        [~, name, ~] = fileparts(filePaths{i});
        jsonMap(name) = char(YPredTest(i));
    end

    jsonText = jsonencode(jsonMap);

    % même nom que modèle, mais en .json
    outFile = erase(matFile, ".mat") + ".json";

    fid = fopen(outFile, "w");
    fwrite(fid, jsonText, "char");
    fclose(fid);

    fprintf("JSON généré : %s\n", outFile);
end

fprintf("\nTous les modèles ont été traités avec succès.\n");
