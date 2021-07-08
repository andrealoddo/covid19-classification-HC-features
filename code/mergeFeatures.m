nameArray=["googleNet","inceptionv3","mobilenetv2","restnet18","restnet50","restnet101","vgg19","alexnet","shufflenet","vgg16"];
%lavoro con ensemble Begged trees
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

load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\train\TOT_ALL_train_table.mat');
tradTrain=trainTable;
load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test\TOT_ALL_test_table.mat');
tradTest=testingTable;

for i=1:length(nameArray)
    load(strcat(rootPath,"\FeaturesCnn\Test\",nameArray(i),"_Plus_ALL_Traditional_Features_Test.mat"));
    testingTableIteration=testingTable;
    testingTableIteration=table(testingTableIteration.Var1,horzcat(testingTableIteration.Var2,tradTest.features));
    testingTableIteration.Properties.VariableNames = {'labels','features'};
    filename=strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCNNHC\test\",nameArray(i),"_HC_Test");
    save(filename,'testingTableIteration','-v7.3');
    
end

