function [trainedClassifier, validationAccuracy] = rusBoostedTrees(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)

inputTable = trainingData;
% Split matrices in the input table into vectors
%parte di auto rilevazione labels e nomi
feature_predictor_names=[];
isCategoricalEdit=[];
[~,dimy]=size(inputTable.features);
for index=1:dimy
    feature_predictor_names=[feature_predictor_names,strcat("features_",string(index))];
    isCategoricalEdit=[isCategoricalEdit,false];
end
isCategoricalEdit=logical(isCategoricalEdit);
feature_predictor_names=cellstr(feature_predictor_names);
%

inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames',feature_predictor_names)];

predictorNames =feature_predictor_names;
predictors = inputTable(:, predictorNames);
response = inputTable.labels;
isCategoricalPredictor = isCategoricalEdit;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical({'covid-19'; 'normal'; 'pneumonia'}));

% Create the result struct with predict function
splitMatricesInTableFcn = @(t) [t(:,setdiff(t.Properties.VariableNames, {'features'})), array2table(table2array(t(:,{'features'})), 'VariableNames',feature_predictor_names)];
extractPredictorsFromTableFcn = @(t) t(:, predictorNames);
predictorExtractionFcn = @(x) extractPredictorsFromTableFcn(splitMatricesInTableFcn(x));
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'features'};
trainedClassifier.ClassificationTree = classificationTree;

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
% Split matrices in the input table into vectors
inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames',feature_predictor_names)];

predictorNames =feature_predictor_names;
predictors = inputTable(:, predictorNames);
response = inputTable.labels;
isCategoricalPredictor = isCategoricalEdit;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
