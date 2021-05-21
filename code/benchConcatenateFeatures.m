function [testingTable] = benchConcatenateFeatures()
%CONCATENATEFEATURES 80-10-10 with multiple features
%   Detailed explanation goes here






%per benchmark 256 imgs
imgArrayLowerL=splitEachLabel(imageDatastore('D:\Tesi\DataInc\train','IncludeSubFolders',true','LabelSource','foldernames'),0.00075);
arrayDescriptors = {'HM',...
                    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6','ZM_10_8','ZM_10_10',...
                    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
                    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
                    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
                    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
                    'HARri','LBP18'};     


testingTable=table();


for ii = 1:length(arrayDescriptors)
    imgArrayLowerL=splitEachLabel(imageDatastore('D:\Tesi\DataInc\train','IncludeSubFolders',true','LabelSource','foldernames'),0.00075);
    
    fprintf('%s%s\n', "Computazione descrittore ",arrayDescriptors{ii});
    fprintf('%s%d%s\n', "Computazione descrittore ",length(arrayDescriptors)-ii," features rimaneti");
    %calcolo delle features

    testingTableInner=extractTableFromImgDatastore(imgArrayLowerL,arrayDescriptors{ii});
    %concatenazione
    testingTable=testingTableInner;
    save(strcat("D:\Tesi\FeaturesSingleCpu\test\",arrayDescriptors(ii),"_test_table.mat"),'testingTable'); 
end
    
end






