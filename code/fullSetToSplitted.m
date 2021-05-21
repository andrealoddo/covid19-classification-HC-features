function [trainTableOut,testTableOut,validateTableOut] = fullSetToSplitted(features,trainTable,testingTable,validateTable)
%FULLSETTOSPLITTED Summary of this function goes here
%   Detailed explanation goes here

%carico o genero l'array delle posizioni e labels
if (nargin==1)
    %se li ho già calcolati li carico nella cartella
   
    load('pre_cmp_positions/positionTest.mat');
    load('pre_cmp_positions/positionValidate.mat');
    load('pre_cmp_positions/positionTrain.mat');
    
    load('pre_cmp_positions/labelsTest.mat');
    load('pre_cmp_positions/labelsValidate.mat');
    load('pre_cmp_positions/labelsTrain.mat');
else
    %altrimenti li creo e salvo nella cartella
    fprintf('%s\n', "Sto calcolando l'array delle posizioni per il test set ");
    positionTest=zeros(height(testingTable),1);
    for n=1:height(testingTable)
        [~,LocResult] = ismembertol(table2array(testingTable(n,'features')),features,'ByRows',true);
        positionTest(n,1)=LocResult;
    end

    fprintf('%s\n', "Sto calcolando l'array delle posizioni per il validation set ");
    positionValidate=zeros(height(validateTable),1);
    for n=1:height(validateTable)
        [~,LocResult] = ismembertol(table2array(validateTable(n,'features')),features,'ByRows',true);
        positionValidate(n,1)=LocResult;
    end

    fprintf('%s\n', "Sto calcolando l'array delle posizioni per il train set ");
    positionTrain=zeros(height(trainTable),1);
    for n=1:height(trainTable)
        [~,LocResult] = ismembertol((table2array(trainTable(n,'features'))),(features),'ByRows',true);
        positionTrain(n,1)=LocResult;
    end
    
    labelsTest=testingTable.labels;
    labelsValidate=validateTable.labels;
    labelsTrain=trainTable.labels;
    if( not(exist('pre_cmp_positions', 'dir') ))
        mkdir('pre_cmp_positions');
    end
    %inizio i salvataggi
    %+ per le posizioni
    save('pre_cmp_positions/positionTest.mat','positionTest');
    save('pre_cmp_positions/positionValidate.mat','positionValidate');
    save('pre_cmp_positions/positionTrain.mat','positionTrain');
    %+ per le labels
    save('pre_cmp_positions/labelsTest.mat','labelsTest');
    save('pre_cmp_positions/labelsValidate.mat','labelsValidate');
    save('pre_cmp_positions/labelsTrain.mat','labelsTrain');
end

%Ricostruisco le tabelle tenendo conto delle posizioni calcolate e di
%eventuali "buchi"
fprintf('%s\n', "Sto calcolando la tabella per il test set ");
validPos=0;
testLabels=labelsTest;
[~,dimy]=size(features);
concFeatures=zeros(height(positionTest),dimy);
for  n=1:height(positionTest)
    if not(positionTest(n)==0)
        validPos=positionTest(n);
    else
        %buchi in caso di resize dell'immagine
        validPos=validPos+1;
    end
    concFeatures(n,:)=features(validPos,:);
    
end    
testTableOut=table(testLabels,concFeatures,'VariableNames',{'labels','features'});

fprintf('%s\n', "Sto calcolando la tabella per il validation set ");
validPos=0;
validateLabels=labelsValidate;
[~,dimy]=size(features);
concFeatures=zeros(height(positionValidate),dimy);
for  n=1:height(positionValidate)
    if not(positionValidate(n)==0)
        validPos=positionValidate(n);
    else
        %buchi in caso di resize dell'immagine
        validPos=validPos+1;
    end
    concFeatures(n,:)=features(validPos,:);
    
end    
validateTableOut=table(validateLabels,concFeatures,'VariableNames',{'labels','features'});

fprintf('%s\n', "Sto calcolando la tabella per il train set ");
validPos=0;
trainLabels=labelsTrain;
[~,dimy]=size(features);
concFeatures=zeros(height(positionTrain),dimy);
for  n=1:height(positionTrain)
    if not(positionTrain(n)==0)
        validPos=positionTrain(n);
    else
        %buchi in caso di resize dell'immagine
        validPos=validPos+1;
    end
    concFeatures(n,:)=features(validPos,:);
    
end    
trainTableOut=table(trainLabels,concFeatures,'VariableNames',{'labels','features'});


end

