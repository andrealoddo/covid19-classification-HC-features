function [trainTable,validateTable] = extractFeaturesAndSplit(descriptor,percentage)
%EXTRACTFEATURESANDSPLIT Suddivide il dataset nella porzione per il train e
%la valutazione, estra le features e prepara i dati


%   Detailed explanation goes here 

%TODO modularizzare al meglio
color = 'gray';
graylevel = 256;
prepro = 'none';
%tramite imageDatastore sfrutto al meglio il dataset assegando le etichette
%corrette
imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
%lo divido in due parti a seconda della percentuale scelta
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArray,percentage);
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
%eseguo i passaggi descritti in precedenza per la porzione di validazione
labels=[];
features=[];
while hasdata(imgArrayLower)
    [element,info]=read(imgArrayLower);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
end
validateTable=table(labels,features);


end

