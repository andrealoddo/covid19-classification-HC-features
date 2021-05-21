function [trainTable,validateTable,testingTable] = concatenateFeatures()
%CONCATENATEFEATURES 80-10-10 with multiple features
%   Detailed explanation goes here

arrayDescriptors = {
                    'ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6','ZM_10_8','ZM_10_10'
                   };       










for ii = 1:length(arrayDescriptors)
    %nuovo per dataset unito
imgArrayUpper=imageDatastore('D:\Tesi\DataInc\train','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayUpperL=imageDatastore('D:\Tesi\DataInc\validate','IncludeSubFolders',true','LabelSource','foldernames');
imgArrayLowerL=imageDatastore('D:\Tesi\DataInc\test','IncludeSubFolders',true','LabelSource','foldernames');

       
%imgArrayUpper=shuffle(imgArrayUpper);
%imgArrayUpperL=shuffle(imgArrayUpperL);
%imgArrayLowerL=shuffle(imgArrayLowerL);

trainTable=table();
validateTable=table();
testingTable=table();
    fprintf('%s%s\n', "Computazione descrittore ",arrayDescriptors{ii});
    fprintf('%s%d%s\n', "Computazione descrittore ",length(arrayDescriptors)-ii," features rimanenti");
    %calcolo delle features
    trainTableInner=extractTableFromImgDatastore(imgArrayUpper,arrayDescriptors{ii});
    validateTableInner=extractTableFromImgDatastore(imgArrayUpperL,arrayDescriptors{ii});
    testingTableInner=extractTableFromImgDatastore(imgArrayLowerL,arrayDescriptors{ii});
    %concatenazione
   
    trainTable=trainTableInner;
    validateTable=validateTableInner;
    testingTable=testingTableInner;
    
    
    save(strcat("D:\Tesi\FeaturesSingleCpu\test\",arrayDescriptors(ii),"_test_table.mat"),'testingTable'); 
    save(strcat("D:\Tesi\FeaturesSingleCpu\train\",arrayDescriptors(ii),"_train_table.mat"),'trainTable');
    save(strcat("D:\Tesi\FeaturesSingleCpu\validate\",arrayDescriptors(ii),"_validate_table.mat"),'validateTable');
  
end






%creo la tabella per il train con colonna delle labels e colonne features
%per poter utilizzare il classification learner


%eseguo i passaggi descritti in precedenza per la porzione di validazione e
%testing

end

