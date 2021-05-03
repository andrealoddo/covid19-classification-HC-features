function [trainTable,testingTable] = extractFeaturesAndSplit(descriptor,percentage)
%EXTRACTFEATURESANDSPLIT Suddivide il dataset nella porzione per il train e
%la valutazione, estra le features e prepara i dati


%   Detailed explanation goes here 


%tramite imageDatastore sfrutto al meglio il dataset assegando le etichette
%corrette
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
imgArray=shuffle(imgArray);
%lo divido in due parti a seconda della percentuale scelta
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArray,percentage);

%creo la tabella per il train con colonna delle labels e colonne features
%per poter utilizzare il classification learner
trainTable=extractTableFromImgDatastore(imgArrayUpper,descriptor);
%eseguo i passaggi descritti in precedenza per la porzione di testing
testingTable=extractTableFromImgDatastore(imgArrayLower,descriptor);


end

