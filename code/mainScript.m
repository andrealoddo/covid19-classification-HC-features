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


%Concateno gli array di features orizzontamente poichè trovo più immediato l'accesso
%----------%
color = 'gray';
graylevel = 256;
prepro = 'none';

%Processo per il Local binary pattern
descriptor = 'LBP18'; 
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');

featuresLBP18=[];
while hasdata(imgArray)
    featuresLBP18=cat(2,featuresLBP18,featureExtraction(read(imgArray), descriptor, color, graylevel, prepro));
end
%Processo per il HAR
descriptor = 'HAR'; 
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
featuresHAR=[];
while hasdata(imgArray)
    featuresHAR=cat(2,featuresHAR,featureExtraction(read(imgArray), descriptor, color, graylevel, prepro));
end

