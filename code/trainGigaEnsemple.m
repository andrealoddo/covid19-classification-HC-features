
%myDir = 'D:\Tesi\covid19-classification-HC-features\code\trainedClassifiers'; %gets directory

myDirFeaturesTrain = 'D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Train'; %gets directory
myFilesFeaturesTrain = dir(fullfile(myDirFeaturesTrain,'\*.mat')); %gets all wav files in struct
myDirFeaturesTest = 'D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Test'; %gets directory
myFilesFeaturesTest = dir(fullfile(myDirFeaturesTest,'\*.mat')); %gets all wav files in struct
if ispc
    % Windows dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    featuresPath = '\FeaturesCnn\';
    ensemblePath='\classifiersCode\';
    classifiersPath='TrainedMaxiFeaturesEnsembles';
    
    % WS dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    featuresPath = '\FeaturesCnn\';
    ensemblePath='\classifiersCode\';
    classifiersPath='TrainedMaxiFeaturesEnsembles';
end


for k = 1:10
    
    baseFileNameFeaturesTrain = myFilesFeaturesTrain(k).name;
    fullFileNameFeaturesTrain = fullfile(baseFileNameFeaturesTrain);
    baseFileNameFeaturesTest = myFilesFeaturesTest(k).name;
    fullFileNameFeaturesTest = fullfile(baseFileNameFeaturesTest);
    fprintf('%s%s\n', "Inizio ");
    load(fullFileNameFeaturesTrain);
    %load(strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test\",classifier.usedFeature,"_test_table.mat"));
    load(fullFileNameFeaturesTest);
    % all of your actions for filtering and plotting go here
    
    trainTable.Properties.VariableNames{'Var1'} = 'labels';
    trainTable.Properties.VariableNames{'Var2'} = 'features';
    testingTable.Properties.VariableNames{'Var1'} = 'labels';
    testingTable.Properties.VariableNames{'Var2'} = 'features';
    if(k==1)
        bigtableTrain=table(trainTable.labels,trainTable.features);
        bigtableTesting=table(testingTable.labels,testingTable.features);
        
   bigtableTrain.Properties.VariableNames{'Var1'} = 'labels';
    bigtableTrain.Properties.VariableNames{'Var2'} = 'features';
    bigtableTesting.Properties.VariableNames{'Var1'} = 'labels';
    bigtableTesting.Properties.VariableNames{'Var2'} = 'features';
    else
        bigtableTrain=table(bigtableTrain.labels,horzcat(bigtableTrain.features,trainTable.features));
        bigtableTesting=table(bigtableTesting.labels,horzcat(bigtableTesting.features,testingTable.features));
        
   bigtableTrain.Properties.VariableNames{'Var1'} = 'labels';
    bigtableTrain.Properties.VariableNames{'Var2'} = 'features';
    bigtableTesting.Properties.VariableNames{'Var1'} = 'labels';
    bigtableTesting.Properties.VariableNames{'Var2'} = 'features';
    end
    
    
end
bigtableTrain.Properties.VariableNames{'Var1'} = 'labels';
bigtableTrain.Properties.VariableNames{'Var2'} = 'features';
bigtableTesting.Properties.VariableNames{'Var1'} = 'labels';
bigtableTesting.Properties.VariableNames{'Var2'} = 'features';
timeinit=tic;
[trainedClassifier,~]=ensembleBeggedTrees(trainTable);
trainedClassifier.classifierType="ensembleBeggedTrees";
labelPredicted=calcTestResults(trainedClassifier,testingTable);
report=calcAccuracy(testingTable.labels,labelPredicted);
trainedClassifier.testedAccuracy=report;
classifier=trainedClassifier;
timefin=toc(timeinit);
classifier.trainingTimeSeconds=timefin;
save(fullfile(rootPath,classifiersPath, "GigaEnsemble"), 'classifier','-v7.3');



%calcolo i risultati delle predizioni
function [results]=calcTestResults(classifier,testTable)
results = classifier.predictFcn(testTable);
end
%calcolo l'accuratezza
function [accuracy]=calcAccuracy(labelOrigin,labelPredicted)
dim=length(labelOrigin);
accuracy=0.;
for i=1:dim
    if labelOrigin(i)==labelPredicted(i)
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/dim;
end
