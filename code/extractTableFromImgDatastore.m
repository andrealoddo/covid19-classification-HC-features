function [outputTable] = extractTableFromImgDatastore(inputDatastore,descriptor)
%EXTRACTTABLEFROMIMGDATASTORE Create ml table from dataset
%   Detailed explanation goes here
%creo array di laber e features che serviranno per comporre la tabella
labels=[];
features=[];
color = 'gray';
graylevel = 256;
prepro = 'none';
%popolo gli array di features e labels concatenando ad ogni iterazione
while hasdata(inputDatastore)
    [element,info]=read(inputDatastore);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
end
outputTable=table(labels,features);
end

