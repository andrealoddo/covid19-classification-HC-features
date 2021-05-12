function [trainTable,validateTable,testingTable] = concatenateFeatures()
%CONCATENATEFEATURES 80-10-10 with multiple features
%   Detailed explanation goes here
bigportion=0.8;
smallportions=0.5;
arrayDescriptors= {'HM',...
                    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6','ZM_10_8','ZM_10_10',...
                    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
                    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
                    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
                    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
                    'HARri','LBP18'};    
arrayDescriptors1= {'HAR','LBP18'};
arrayDescriptors3= {'HM','HAR','LBP18',};






%nuovo per dataset unito
imgArrayUpper=imageDatastore('D:\Tesi\DataInc\train','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayUpperL=imageDatastore('D:\Tesi\DataInc\validate','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayLowerL=imageDatastore('D:\Tesi\DataInc\test','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayUpper=shuffle(imgArrayUpper);
imgArrayUpperL=shuffle(imgArrayUpperL);
imgArrayLowerL=shuffle(imgArrayLowerL);

trainTable=table();
validateTable=table();
testingTable=table();


for ii = 1:length(arrayDescriptors)
    fprintf('%s%s\n', "Computazione descrittore ",arrayDescriptors{ii});
    fprintf('%s%d%s\n', "Computazione descrittore ",length(arrayDescriptors)-ii," features rimaneti");
    %calcolo delle features
    trainTableInner=extractTableFromImgDatastore(imgArrayUpper,arrayDescriptors{ii});
    validateTableInner=extractTableFromImgDatastore(imgArrayUpperL,arrayDescriptors{ii});
    testingTableInner=extractTableFromImgDatastore(imgArrayLowerL,arrayDescriptors{ii});
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

