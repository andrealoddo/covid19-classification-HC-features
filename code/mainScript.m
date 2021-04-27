% Esempio di calcolo delle feature per una singola immagine
addpath(genpath('featuresComputation'));
addpath(genpath('HAR'));
img = imread('../img.png');

descriptor = 'LBP18'; % 'HAR', o 'LBP18'
color = 'gray';
graylevel = 256;
prepro = 'none';

features = featureExtraction(img, descriptor, color, graylevel, prepro);

%Esempio di calcolo features per piu' immagini:
%img = datastore(....) o funzioni simili

%for i = 1:size(img)
    
    %features = [features; featureExtraction(img, descriptor, color, graylevel, prepro)'];
    
%end
