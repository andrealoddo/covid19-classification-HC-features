function [trainedClassifier, validationAccuracy] = ensembleBeggedTrees(trainingData)
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

%
% Split matrices in the input table into vectors

inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames', feature_predictor_names)];

predictorNames = feature_predictor_names;
predictors = inputTable(:, predictorNames);
response = inputTable.labels;
isCategoricalPredictor = isCategoricalEdit;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 143777);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', categorical({'covid-19'; 'normal'; 'pneumonia'}));

% Create the result struct with predict function
splitMatricesInTableFcn = @(t) [t(:,setdiff(t.Properties.VariableNames, {'features'})), array2table(table2array(t(:,{'features'})), 'VariableNames', feature_predictor_names)];
extractPredictorsFromTableFcn = @(t) t(:, predictorNames);
predictorExtractionFcn = @(x) extractPredictorsFromTableFcn(splitMatricesInTableFcn(x));
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'features'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
% Split matrices in the input table into vectors
inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'features'})), array2table(table2array(inputTable(:,{'features'})), 'VariableNames', feature_predictor_names)];

predictorNames = feature_predictor_names;
predictors = inputTable(:, predictorNames);
response = inputTable.labels;
isCategoricalPredictor = isCategoricalEdit;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 10);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
