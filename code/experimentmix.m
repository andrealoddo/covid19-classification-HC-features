%features=trainTable.Var2;
%labels=trainTable.Var1;
%features=testingTable.Var2;
%labels=testingTable.Var1;
%features=horzcat(features,trainTable.Var2);
%trainTable=table(labels,features);
%features=horzcat(features,testingTable.Var2);
%testingTable=table(labels,features);
%save(strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Train\","vgg9shufflenet_Features_Train"),'trainTable','-v7.3');
%save(strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Test\","vgg9shufflenet_Features_Test"),'testingTable','-v7.3');


myDir = 'D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Train'; %gets directory
myFiles = dir(fullfile(myDir,'*.mat')); %gets all wav files in struct
myDirFeatures = 'D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Test'; %gets directory
myFilesFeatures = dir(fullfile(myDirFeatures,'*.mat')); %gets all wav files in struct

for k = 5:length(myFiles)
    
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(baseFileName);
    baseFileNameFeatures = myFilesFeatures(k).name;
    fullFileNameFeatures = fullfile(baseFileNameFeatures);
    load(baseFileNameFeatures);
    load(fullFileName);
    trainTable.Properties.VariableNames{'Var1'} = 'labels';
    trainTable.Properties.VariableNames{'Var2'} = 'features';
    testingTable.Properties.VariableNames{'Var1'} = 'labels';
    testingTable.Properties.VariableNames{'Var2'} = 'features';
    
    if  k==5
        fprintf('%s%s\n', "unisco",fullFileName);
        trainTableIter=trainTable;
        testingTableIter=testingTable;
    elseif k==9 || k==6 || k==7
        fprintf('%s%s\n', "NONunisco",fullFileName);
    else
        fprintf('%s%s\n', "unisco",fullFileName);
        testingTableIter=table(testingTableIter.labels,horzcat(testingTableIter.features,testingTable.features));
        trainTableIter=table(trainTableIter.labels,horzcat(trainTableIter.features,trainTable.features));
        trainTableIter.Properties.VariableNames{'Var1'} = 'labels';
        trainTableIter.Properties.VariableNames{'Var2'} = 'features';
        testingTableIter.Properties.VariableNames{'Var1'} = 'labels';
        testingTableIter.Properties.VariableNames{'Var2'} = 'features';
        
    end
    
end
