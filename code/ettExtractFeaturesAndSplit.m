function [trainTable,validateTable,testingTable] = ettExtractFeaturesAndSplit(descriptor)
%ETTEXTRACTFEATURESANDSPLIT Summary of this function goes here
%   Detailed explanation goes here

%TODO modularizzare al meglio
color = 'gray';
graylevel = 256;
prepro = 'none';
bigportion=0.8;
smallportions=0.5;
%tramite imageDatastore sfrutto al meglio il dataset assegando le etichette
%corrette
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
imgArray.shuffle;
%lo divido in due parti a seconda della percentuale scelta
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArray,bigportion);
%creo array di laber e features che serviranno per comporre la tabella
labels=[];
features=[];
%popolo gli array di features e labels concatenando ad ogni iterazione
while hasdata(imgArrayUpper)
    [element,info]=read(imgArrayUpper);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
end
%creo la tabella per il train con colonna delle labels e colonne features
%per poter utilizzare il classification learner
trainTable=table(labels,features);
%splitto in due la lower
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArrayLower,smallportions);
%eseguo i passaggi descritti in precedenza per la porzione di validazione e
%testing
labels=[];
features=[];

while hasdata(imgArrayLower)
    [element,info]=read(imgArrayLower);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
end
validateTable=table(labels,features);

labels=[];
features=[];
while hasdata(imgArrayUpper)
    [element,info]=read(imgArrayUpper);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
end
testingTable=table(labels,features);
end
