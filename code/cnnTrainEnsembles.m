function [] = cnnTrainEnsembles()

if ispc
    % Windows dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    featuresPath = '\FeaturesCNNHC\';
    ensemblePath='\classifiersCode\';
    classifiersPath='TrainedMaxiFeaturesEnsembles';
    
    % WS dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    featuresPath = '\FeaturesCNNHC\';
    ensemblePath='\classifiersCode\';
    classifiersPath='TrainedMaxiFeaturesEnsembles';
end
%calcolo i risultati delle predizioni
function [results]=calcTestResults(classifier,testTable)
results = classifier.predictFcn(testTable);
end
%calcolo l'accuratezza
function [accuracy]=calcAccuracy(labelOrigin,labelPredicted)
dim=length(labelOrigin);
accuracy=0.;
for isw=1:dim
    if labelOrigin(isw)==labelPredicted(isw)
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/dim;
end

fileArray=["googlenet__EP20__MBS32.mat",...
    "inceptionv3__EP20__MBS32.mat",...
    "mobilenetv2__EP20__MBS32.mat",...
    "resnet18__EP20__MBS32.mat",...
    "resnet50__EP20__MBS32.mat",...
    "resnet101__EP20__MBS32.mat",...
    "vgg19__EP20__MBS32.mat",...
    "alexnet__EP20__MBS32",...
    "shufflenet__EP20__MBS32.mat",...
    "vgg16__EP20__MBS32.mat"
    ];
nameArrayOrig=["googleNet","inceptionv3","mobilenetv2","restnet18","restnet50","restnet101","vgg19","alexnet","shufflenet","vgg16"];
nameArray=["inceptionv3","mobilenetv2","restnet18","restnet50","restnet101","vgg19","alexnet","shufflenet","vgg16"];
%lavoro con ensemble Begged trees

for isw=1:length(nameArray)

     classifier = "ensembleBoostedTrees";
    trainTable=load(strcat(rootPath,"\FeaturesCNNHC\train\",nameArray(isw),"_HC_Train.mat")).trainTableIteration;
    testingTable=load(strcat(rootPath,"\FeaturesCNNHC\test\",nameArray(isw),"_HC_Test.mat")).testingTableIteration;
    %trainTable=head(trainTable,10);
    %testingTable=head(testingTable,10);

    filename=strcat('trained_', "ensembleBoostedTrees", '_HC_', nameArray(isw));
    if( exist( fullfile( rootPath,classifiersPath, classifier, strcat(filename, '.mat') , 'file')) == 2 )
        fprintf('%s\n', "ensembleBoostedTrees esiste");
    else
        timeinit=tic;
        fprintf('%s\n', "Computazione classificatori per ensemble Boosted trees");
        %trainTable.Properties.VariableNames{'Var1'} = 'labels';
        %trainTable.Properties.VariableNames{'Var2'} = 'features';
        %testingTable.Properties.VariableNames{'Var1'} = 'labels';
        %testingTable.Properties.VariableNames{'Var2'} = 'features';
        [trainedClassifier,~]=ensembleBoostedTrees(trainTable);
        trainedClassifier.classifierType="ensembleBeggedTrees";
        labelPredicted=calcTestResults(trainedClassifier,testingTable);
        report=calcAccuracy(testingTable.labels,labelPredicted);
        trainedClassifier.testedAccuracy=report;
        classifier=trainedClassifier;
        timefin=toc(timeinit);
        classifier.trainingTimeSeconds=timefin;
        save(fullfile(rootPath,classifiersPath, filename), 'classifier','-v7.3');
    end
    
 
end
end

