function [trainTable,validateTable,testingTable] = concatenateFeatures()
%CONCATENATEFEATURES 80-10-10 with multiple features
%   Detailed explanation goes here
bigportion=0.8;
smallportions=0.5;
arrayDescriptors=["HAR","LBP18"];

imgArray=imageDatastore('../dataset_subset/','IncludeSubFolders',true','LabelSource','foldernames');
imgArray=shuffle(imgArray);
%lo divido in due parti a seconda della percentuale scelta
[imgArrayUpper,imgArrayLower]=splitEachLabel(imgArray,bigportion);
%splitto in due la lower
[imgArrayUpperL,imgArrayLowerL]=splitEachLabel(imgArrayLower,smallportions);
trainTable=table();
validateTable=table();
testingTable=table();


for ii = 1:length(arrayDescriptors)
    %calcolo delle features
    trainTableInner=extractTableFromImgDatastore(imgArrayUpper,arrayDescriptors(ii));
    validateTableInner=extractTableFromImgDatastore(imgArrayUpperL,arrayDescriptors(ii));
    testingTableInner=extractTableFromImgDatastore(imgArrayLowerL,arrayDescriptors(ii));
    %concatenazione
    if isempty(trainTable)
        trainTable=trainTableInner;
        validateTable=validateTableInner;
        testingTable=testingTableInner;
        %rinomina perchè altrimenti come varname esce labels(è il nome dato all'array nella funzione per l'estrazione della table)
        trainTable = renamevars(trainTable,'features',arrayDescriptors(ii));
        validateTable = renamevars(validateTable,'features',arrayDescriptors(ii));
        testingTable = renamevars(testingTable,'features',arrayDescriptors(ii));
    else
        %do come nome alla var il descrittore
        %TODO creazione array con nomi più esplicativi esempio features
        %+" "+descrittore
        trainTable=addvars(trainTable,trainTableInner.features,'NewVariableNames',arrayDescriptors(ii));
        validateTable=addvars(validateTable,validateTableInner.features,'NewVariableNames',arrayDescriptors(ii));
        testingTable=addvars(testingTable,testingTableInner.features,'NewVariableNames',arrayDescriptors(ii)); 
    end
end






%creo la tabella per il train con colonna delle labels e colonne features
%per poter utilizzare il classification learner


%eseguo i passaggi descritti in precedenza per la porzione di validazione e
%testing

end

