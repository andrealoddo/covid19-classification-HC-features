function [trainTable,validateTable,testingTable] = ettExtractFeaturesAndSplit(descriptor)
%ETTEXTRACTFEATURESANDSPLIT Summary of this function goes here
%   Detailed explanation goes here
bigportion=0.8;
smallportions=0.5;
%tramite imageDatastore sfrutto al meglio il dataset assegando le etichette
%corrette
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
imgArray=shuffle(imgArray);
%lo divido in due parti a seconda della percentuale scelta
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArray,bigportion);

%creo la tabella per il train con colonna delle labels e colonne features
%per poter utilizzare il classification learner
trainTable=extractTableFromImgDatastore(imgArrayUpper,descriptor);
%splitto in due la lower
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArrayLower,smallportions);
%eseguo i passaggi descritti in precedenza per la porzione di validazione e
%testing
validateTable=extractTableFromImgDatastore(imgArrayUpper,descriptor);

testingTable=extractTableFromImgDatastore(imgArrayLower,descriptor);
end
