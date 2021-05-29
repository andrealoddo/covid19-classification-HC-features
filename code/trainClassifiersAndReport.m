function [] = trainClassifiersAndReport()

addpath(genpath('trainedClassifiers'));
addpath("classifiersCode");

descriptors_sets = {'HM',...
    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6',...
    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
    'HARri','LBP18'};

%se non posso usare il parallel toolbox
if (isempty(ver('parallel')))
    for index=1:length(descriptors_sets)
        fprintf('%s%s\n', "Computazione classificatori per ",descriptors_sets{index});
        extractTrainedClassifiersFromFeature(descriptors_sets{index});
    end
else
    parfor index=1:length(descriptors_sets)
        fprintf('%s%s\n', "Computazione classificatori per ",descriptors_sets{index});
        extractTrainedClassifiersFromFeature(descriptors_sets{index});
    end
end

end

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
%estraggo l'accuratezza (train) per un claddificatore
function [classifier]=extractTrainedClassifiersFromFeature(feature)

if ispc
    % Windows dataset path
    rootPath = 'D:\Tesi';
    featuresPath = 'code\FeaturesSingleCpu';
    classifiersPath = 'trainedClassifiers';
else
    % WS dataset path
    rootPath = '/home/server/MATLAB/topics/covid19-classification-HC-features/code';
    featuresPath = 'FeaturesSingleCpu';
    classifiersPath = 'trainedClassifiers';
end
%ogni volta ricarico per certezza che non vengano consumati

trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;

%lavoro con rus Boosted trees
classifier = "rusBoostedTrees";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "rus Boosted trees esiste");
else
    timeinit=tic;
    [trainedClassifier,~]=rusBoostedTrees(trainTable);
    trainedClassifier.classifierType="rusBoostedTrees";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con il coarse Gaussian svm
classifier = "coarseGaussianSvm";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "coarseGaussianSvm esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per coarseGaussianSvm");
    [trainedClassifier,~]=coarseGaussianSvm(trainTable);
    trainedClassifier.classifierType="coarseGaussianSvm";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con ensemble Begged trees
classifier = "ensembleBeggedTrees";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "ensembleBeggedTrees esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per ensemble Begged trees");
    [trainedClassifier,~]=ensembleBeggedTrees(trainTable);
    trainedClassifier.classifierType="ensembleBeggedTrees";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con fine tree
classifier = "fineTree";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "fineTree esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per fine tree");
    [trainedClassifier,~]=fineTree(trainTable);
    trainedClassifier.classifierType="fineTree";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con gaussianNaiveBayes
classifier = "gaussianNaiveBayes";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "gaussianNaiveBayes esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per gaussianNaiveBayes");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=gaussianNaiveBayes(trainTable);
    trainedClassifier.classifierType="gaussianNaiveBayes";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con kernelNaiveBayes
classifier = "kernelNaiveBayes";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "kernelNaiveBayes esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per kernelNaiveBayes");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=kernelNaiveBayes(trainTable);
    trainedClassifier.classifierType="kernelNaiveBayes";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end

%lavoro con linearSvm
classifier = "linearSvm";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "linearSvm esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per linearSvm");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=linearSvm(trainTable);
    trainedClassifier.classifierType="linearSvm";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con mediumGaussianSvm
classifier = "mediumGaussianSvm";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "mediumGaussianSvm esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per mediumGaussianSvm");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=mediumGaussianSvm(trainTable);
    trainedClassifier.classifierType="mediumGaussianSvm";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con mediumNeuralNetwork
classifier = "mediumNeuralNetwork";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "mediumNeuralNetwork esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per mediumNeuralNetwork");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=mediumNeuralNetwork(trainTable);
    trainedClassifier.classifierType="mediumNeuralNetwork";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con narrowNeuralNetwork
classifier = "narrowNeuralNetwork";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "narrowNeuralNetwork esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per narrowNeuralNetwork");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=narrowNeuralNetwork(trainTable);
    trainedClassifier.classifierType="narrowNeuralNetwork";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con quadraticSvm
classifier = "quadraticSvm";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "quadraticSvm esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per quadraticSvm");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=quadraticSvm(trainTable);
    trainedClassifier.classifierType="quadraticSvm";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end


%lavoro con weightedKnn
classifier = "weightedKnn";

if( exist( fullfile( classifiersPath, classifier, strcat('trained_', ...
        classifier, '_', feature, '.mat') ), 'file') == 2 )
    fprintf('%s\n', "weightedKnn esiste");
else
    timeinit=tic;
    fprintf('%s\n', "Computazione classificatori per weightedKnn");
    trainTable=load(fullfile(rootPath, featuresPath, 'train', strcat(feature, '_train_table.mat'))).trainTable;
    testingTable=load(fullfile(rootPath, featuresPath, 'test', strcat(feature, '_test_table.mat'))).testingTable;
    [trainedClassifier,~]=weightedKnn(trainTable);
    trainedClassifier.classifierType="weightedKnn";
    trainedClassifier.usedFeature=feature;
    labelPredicted=calcTestResults(trainedClassifier,testingTable);
    report=calcAccuracy(testingTable.labels,labelPredicted);
    trainedClassifier.testedAccuracy=report;
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(fullfile(classifiersPath, trainedClassifier.classifierType, strcat('trained_', ...
        trainedClassifier.classifierType, '_', trainedClassifier.usedFeature)), 'classifier');
end

end
