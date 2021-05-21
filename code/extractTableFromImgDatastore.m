function [outputTable] = extractTableFromImgDatastore(inputDatastore,descriptor)
tic
%EXTRACTTABLEFROMIMGDATASTORE Create ml suitable table from dataset
%Tramite calcolo parallelo dimezzo(circa) il tempo necessario
%creo array di laber e features che serviranno per comporre la tabella
labels=[];
features=[];
color = 'gray';
graylevel = 256;
prepro = 'none';
counter=0;

   
    while hasdata(inputDatastore)
        if(mod(counter,10000)==0)
             fprintf('%s%d%s\n', "Feature estratte per ",counter," immagini!");
        end    
        counter=counter+1;
    
        [element,info]=read(inputDatastore);
        if not(size(element)==[512,512]) 
        element=imresize(element,[512,512]);
        end  
        labels=vertcat(labels,info.Label);
        features=vertcat(features,featureExtraction(element, descriptor, color, graylevel, prepro).');
    end

toc
outputTable=table(labels,features);