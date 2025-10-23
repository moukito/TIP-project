%% ===============================================================
%  Train_EfficientNetB0_Food101.m
%  Fine-tuning EfficientNetB0 pour Food-101 (11 classes)
%  A lancer APRES DataLoader.m (qui crée augimdsTrain, augimdsVal, imdsVal)
%% ===============================================================

clc; 
try, gpuDevice(1); catch, warning('Pas de GPU détecté.'); end

% --- Hypers ---
numClasses = 11;
miniBatch  = 32;
maxEpochs1 = 20;
maxEpochs2 = 8;     % petit affinement
lr1        = 3e-4;
lr2        = 1e-5;

% --- Vérifs workspace (on n'édite pas ton DataLoader) ---
assert(exist('augimdsTrain','var')==1, 'augimdsTrain manquant (exécute d''abord DataLoader)');
assert(exist('augimdsVal','var')==1,   'augimdsVal manquant (exécute d''abord DataLoader)');
assert(exist('imdsVal','var')==1,      'imdsVal manquant (exécute d''abord DataLoader)');

% --- Recréer un transform propre (sans toucher à ton fichier) ---
tdsTrainSafe = transform(augimdsTrain, @applyJitterSafe);

% --- Charger EfficientNet-B0 ---
net = efficientnetb0; 
lgraph = layerGraph(net);

% DataLoader normalise déjà (mean/std ImageNet) -> pas de renormalisation ici
% On remplace l'input layer par une version 'Normalization','none'
inpIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ImageInputLayer'), lgraph.Layers), 1, 'first');
if ~isempty(inpIdx)
    oldInp = lgraph.Layers(inpIdx);
    newInp = imageInputLayer(oldInp.InputSize, 'Name', oldInp.Name, 'Normalization','none');
    lgraph = replaceLayer(lgraph, oldInp.Name, newInp);
end

% --- Remplacer la tête de classification de manière robuste (sans hard-code) ---
layers = lgraph.Layers; conns = lgraph.Connections;

% Classification layer (quel que soit son nom)
idxCls = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1, 'last');
assert(~isempty(idxCls), 'Aucune ClassificationOutputLayer trouvée.');
clsName = layers(idxCls).Name;

% FullyConnected finale (si présente)
isFC = arrayfun(@(L) contains(class(L),'FullyConnectedLayer'), layers);
idxFC = find(isFC, 1, 'last');

if ~isempty(idxFC)
    fcName = layers(idxFC).Name;
    lgraph = replaceLayer(lgraph, fcName, ...
        fullyConnectedLayer(numClasses,'Name',fcName,'WeightsInitializer','he','BiasInitializer','zeros'));
    lgraph = replaceLayer(lgraph, clsName, classificationLayer('Name', clsName));
else
    % Cas rare: pas de FC explicite. On insère notre tête après le GAP.
    % On trouve le prédécesseur de la classification:
    prevToCls = conns.Source(strcmp(conns.Destination, clsName));
    assert(~isempty(prevToCls), 'Impossible de trouver la couche précédente à %s.', clsName);
    prevName = prevToCls{1};

    lgraph = removeLayers(lgraph, {clsName});
    newHead = [
        dropoutLayer(0.3,'Name','new_dropout')
        fullyConnectedLayer(numClasses,'Name','new_fc','WeightsInitializer','he','BiasInitializer','zeros')
        softmaxLayer('Name','new_softmax')
        classificationLayer('Name','new_output') ];
    lgraph = addLayers(lgraph, newHead);
    lgraph = connectLayers(lgraph, prevName, 'new_dropout');
end

% --- Options Phase 1 ---
opts1 = trainingOptions('adam', ...
    'MiniBatchSize', miniBatch, ...
    'MaxEpochs', maxEpochs1, ...
    'InitialLearnRate', lr1, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 6, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation');

% --- Train Phase 1 (sur tdsTrainSafe qui gère la casse input/Input) ---
fprintf('\n=== Phase 1: EfficientNetB0 ===\n');
[net1, info1] = trainNetwork(tdsTrainSafe, lgraph, opts1);

% --- Options Phase 2 (affinage fin) ---
opts2 = trainingOptions('adam', ...
    'MiniBatchSize', miniBatch, ...
    'MaxEpochs', maxEpochs2, ...
    'InitialLearnRate', lr2, ...
    'L2Regularization', 5e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 8, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'OutputNetwork', 'best-validation');

fprintf('\n=== Phase 2: Affinage fin ===\n');
[trainedNet, info2] = trainNetwork(tdsTrainSafe, net1.Layers, opts2);

% --- Sauvegarde ---
if ~isfolder('models'), mkdir models; end
save(fullfile('models','EfficientNetB0_Food101.mat'),'trainedNet','info1','info2','-v7.3');

% --- Évaluation Validation ---
[YPred, ~] = classify(trainedNet, augimdsVal, 'MiniBatchSize', miniBatch);
YTrue = imdsVal.Labels;
valAcc = mean(YPred == YTrue)*100;
fprintf('Validation accuracy: %.2f %%\n', valAcc);

figure('Name','Confusion Matrix - Validation');
confusionchart(YTrue, YPred);
title(sprintf('Validation (%.2f%%)', valAcc));

%% ===================== TRANSFORM ROBUSTE =======================
function dataOut = applyJitterSafe(dataIn)
% Compatible avec 'input' ou 'Input' (table), batch ou image.
    dataOut = dataIn;
    vars = dataIn.Properties.VariableNames;

    % Détecter la colonne image (input/Input)
    if any(strcmp(vars,'input'))
        vin = 'input';
    elseif any(strcmp(vars,'Input'))
        vin = 'Input';
    else
        error('Colonne image introuvable. Variables: %s', strjoin(vars, ', '));
    end

    % S'il existe une colonne de labels, on la laisse inchangée
    n = height(dataIn);
    for i = 1:n
        img = dataIn.(vin){i};   % images sont stockées en cellules
        img = im2double(img);

        % === Jitter photométrique (mêmes plages que ton DataLoader) ===
        img = img * (0.7 + rand()*0.6);                   % brightness [0.7,1.3]
        img = imadjust(img, [], [], 0.8 + rand()*0.4);    % gamma     [0.8,1.2]
        hsv = rgb2hsv(img);
        hsv(:,:,2) = hsv(:,:,2) .* (0.8 + rand()*0.4);    % saturation
        img = hsv2rgb(hsv);
        img = min(max(img,0),1);

        % === Normalisation ImageNet (comme ton DataLoader) ===
        mu = reshape([0.485 0.456 0.406],1,1,3);
        sig= reshape([0.229 0.224 0.225],1,1,3);
        img = (img - mu) ./ sig;

        dataOut.(vin){i} = single(img);
    end
end
