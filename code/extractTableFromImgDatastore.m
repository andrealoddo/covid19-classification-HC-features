function [outputTable] = extractTableFromImgDatastore(inputDatastore,descriptor)
%EXTRACTTABLEFROMIMGDATASTORE Create ml suitable table from dataset
%Tramite calcolo parallelo dimezzo(circa) il tempo necessario
%creo array di laber e features che serviranno per comporre la tabella
labels=[];
features=[];
color = 'gray';
graylevel = 256;
prepro = 'none';
%popolo gli array di features e labels concatenando ad ogni iterazione
%controllo se posso usare il calcolo in parallelo
if not(isempty(ver('parallel')))  
    %suddivido il carico delle partizioni tra più workers
    n = numpartitions(inputDatastore,gcp); 
    %parallel for per utilizzare i diversi workers
    parfor ii = 1:n
        labelsin=[];
        featuresin=[];
        subds = partition(inputDatastore,n,ii);
          while hasdata(subds)
            [element,info]=read(subds);
            %questi due array sono temporanei, è importante che vengano
            %dichiarati all'interno del parfor e ad ogni iterazione vengono
            %reinizializzati
            labelsin=cat(1,labelsin,info.Label');
            featuresin=cat(1,featuresin,featureExtraction(element,descriptor,color, graylevel, prepro).');

          end
          %vertcat unica soluzione non concateva col cat tradizionale
          labels=vertcat(labels,labelsin);
          features=vertcat(features,featuresin);
    end
else  
    %non avendo a disposizione il pct si intera in maniera classica 
    while hasdata(inputDatastore)
    [element,info]=read(inputDatastore);
    labels=cat(1,labels,info.Label);
    features=cat(1,features,featureExtraction(element, descriptor, color, graylevel, prepro).');
    end
end
outputTable=table(labels,features);