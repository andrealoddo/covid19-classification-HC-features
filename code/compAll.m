%dataset
imdsTrain=imageDatastore('D:\Tesi\DataIncRGB\train','IncludeSubFolders',true','LabelSource','foldernames');
imdsValid=imageDatastore('D:\Tesi\DataIncRGB\validate','IncludeSubFolders',true','LabelSource','foldernames');
imdsTest=imageDatastore('D:\Tesi\DataIncRGB\test','IncludeSubFolders',true','LabelSource','foldernames');


%imdsTrain=splitEachLabel(imdsTrain,0.001);
%imdsValid=splitEachLabel(imdsValid,0.001);
%imdsTest=splitEachLabel(imdsTest,0.001);

imdsTrain.ReadFcn = @customReadDatastoreImage;
imdsValid.ReadFcn = @customReadDatastoreImage;
imdsTest.ReadFcn = @customReadDatastoreImage;


% Training options
miniBatchSize = 8;
maxEpochs = 50;
valFrequency = max(floor(numel(imdsTest.Files)/miniBatchSize)*10,1);
netCheckPath="checkpoint";


options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 0.1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValid, ...
    'ValidationFrequency', valFrequency, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'CheckpointPath', netCheckPath, ...
    'OutputFcn', @(info)stopIfAccuracyNotImproving( info, 5 ));

net = trainNetwork(imdsTrain,layerGraph(net),options);


function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = imresize(data,[224 224]);
end