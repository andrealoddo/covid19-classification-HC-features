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


%preparo il dataset per il training
[trainHAR,validateHAR]=extractFeaturesAndSplit('HAR',0.8);
[trainLBP18,validateLBP18]=extractFeaturesAndSplit('LBP18',0.8);
%eseguo il predict tramite il modello 
%essendo il dataset ancora di piccole dimensioni i risultati possono essere
%falsati, aumentando il fold è possibile ottenere risultati più concreti
%prima di eseguire questa porzione è necessario avere un modello già
%allenato
predictions = trainedModel.predictFcn(validateHAR);
predictions1 = trainedModel1.predictFcn(validateHAR);
%calcolo l'accuratezza
iscorrect=predictions1==validateHAR.labels;
iscorrect=iscorrect(:,1);
sizeArray=size(validateHAR);
accuracy=sum(iscorrect)*100/sizeArray(1);

