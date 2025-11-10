% ------------------------------------------------------------
% Génération du fichier JSON
% ------------------------------------------------------------
fprintf('Génération du fichier JSON...\n');

filePaths = imdsTest.Files;
n = numel(filePaths);

% Utilisation d'un containers.Map pour accepter des clés '0', '1', etc.
jsonMap = containers.Map('KeyType','char', 'ValueType','char');

for i = 1:n
    [~, name, ~] = fileparts(filePaths{i});  % ex. '0', '1', '1234'
    jsonMap(name) = char(YPredTest(i));      % ex. 'bread', 'fried'
end

% Encodage en JSON
jsonText = jsonencode(jsonMap);

outFile = 'predictions_lenet5.json';
fid = fopen(outFile,'w');
fwrite(fid, jsonText, 'char');
fclose(fid);

fprintf('Fichier %s généré avec succès !\n', outFile);
