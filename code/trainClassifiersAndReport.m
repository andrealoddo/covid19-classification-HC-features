function [] = trainClassifiersAndReport()
%TRAINCLASSIFIERSANDREPORT Summary of this function goes here
%   Detailed explanation goes here
addpath(genpath('trainedClassifiers'));
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
     %ogni volta ricarico per certezza che non vengano consumati
     
     %lavoro con rus Boosted trees
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per rus Boosted trees");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=rusBoostedTrees(trainTable);
     trainedClassifier.classifierType="rusBoostedTrees";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
     
     %lavoro con il coarse Gaussian svm
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per coarseGaussianSvm");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=coarseGaussianSvm(trainTable);
     trainedClassifier.classifierType="coarseGaussianSvm";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
     
     %lavoro con ensemble Begged trees
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per ensemble Begged trees");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=ensembleBeggedTrees(trainTable);
     trainedClassifier.classifierType="ensembleBeggedTrees";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
          
     %lavoro con fine tree
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per fine tree");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=fineTree(trainTable);
     trainedClassifier.classifierType="fineTree";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
               
     %lavoro con gaussianNaiveBayes
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per gaussianNaiveBayes");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=gaussianNaiveBayes(trainTable);
     trainedClassifier.classifierType="gaussianNaiveBayes";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                    
     %lavoro con kernelNaiveBayes
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per kernelNaiveBayes");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=kernelNaiveBayes(trainTable);
     trainedClassifier.classifierType="kernelNaiveBayes";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
     
                    
     %lavoro con linearSvm
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per linearSvm");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=linearSvm(trainTable);
     trainedClassifier.classifierType="linearSvm";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                         
     %lavoro con mediumGaussianSvm
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per mediumGaussianSvm");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=mediumGaussianSvm(trainTable);
     trainedClassifier.classifierType="mediumGaussianSvm";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                              
     %lavoro con mediumNeuralNetwork
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per mediumNeuralNetwork");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=mediumNeuralNetwork(trainTable);
     trainedClassifier.classifierType="mediumNeuralNetwork";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                                   
     %lavoro con narrowNeuralNetwork
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per narrowNeuralNetwork");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=narrowNeuralNetwork(trainTable);
     trainedClassifier.classifierType="narrowNeuralNetwork";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                                        
     %lavoro con quadraticSvm
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per quadraticSvm");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=quadraticSvm(trainTable);
     trainedClassifier.classifierType="quadraticSvm";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');
                                             
     %lavoro con weightedKnn
     timeinit=tic;
      fprintf('%s\n', "Computazione classificatori per weightedKnn");
     trainTable=load(strcat("D:\Tesi\FeaturesSingleCpu\train\",feature,"_train_table.mat")).trainTable;
     testingTable=load(strcat("D:\Tesi\FeaturesSingleCpu\test\",feature,"_test_table.mat")).testingTable;
     [trainedClassifier,~]=weightedKnn(trainTable);
     trainedClassifier.classifierType="weightedKnn";
     trainedClassifier.usedFeature=feature;
     labelPredicted=calcTestResults(trainedClassifier,testingTable);
     report=calcAccuracy(testingTable.labels,labelPredicted);
     trainedClassifier.testedAccuracy=report;
     classifier=trainedClassifier;
     timefin=toc(timeinit);
     classifier.trainingTimeSeconds=timefin;
     save(strcat("trainedClassifiers/",trainedClassifier.classifierType,"/trained_",...
         trainedClassifier.classifierType,"_", trainedClassifier.usedFeature),'classifier');

end

