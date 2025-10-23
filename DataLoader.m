% --- Dossiers ---
trainFolder = fullfile('dataset', 'train');
testFolder  = fullfile('dataset', 'test');

% --- Créer datastore ---
imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

rng(42);  % pour fixer la graine aléatoire
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');

% --- Data augmentation géométrique ---
augmenter = imageDataAugmenter( ...
    'RandRotation', [-25, 25], ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandXScale', [0.85 1.2], ...
    'RandYScale', [0.85 1.2]);

inputSize = [224 224 3];

% --- Appliquer la géométrie ---
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augimdsVal   = augmentedImageDatastore(inputSize, imdsVal);

% --- Créer un datastore transformé avec jitter photométrique ---
tdsTrain = transform(augimdsTrain, @applyJitter);

% --- Fonction utilitaire pour luminosité / contraste ---
function dataOut = applyJitter(dataIn)
    dataOut = dataIn;
    for i = 1:size(dataIn.Input,4)
        img = dataIn.Input(:,:,:,i);
        img = im2double(img);

        % Brightness
        img = img * (0.7 + rand() * 0.6); % [0.7,1.3]

        % Contrast (gamma)
        img = imadjust(img, [], [], 0.8 + rand() * 0.4); % [0.8,1.2]

        % Color jitter (simulate saturation shifts)
        hsv = rgb2hsv(img);
        hsv(:,:,2) = hsv(:,:,2) .* (0.8 + rand() * 0.4); % saturation
        img = hsv2rgb(hsv);
        img = min(max(img,0),1);

        % Normalisation ImageNet
        img = (img - reshape([0.485 0.456 0.406],1,1,3)) ./ ...
              reshape([0.229 0.224 0.225],1,1,3);

        dataOut.Input(:,:,:,i) = single(img);
    end
end

