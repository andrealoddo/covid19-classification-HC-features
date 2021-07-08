function [] = cnnExtractFeatures()
%CNNEXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here

if ispc
    % Windows dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    classifiersPath = '\PretrainedCNN\';
   
    % WS dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    classifiersPath = '\PretrainedCNN\';
end
%ogni volta ricarico per certezza che non vengano consumati



layerArray = {["pool5-drop_7x7_s1"],...
    ["avg_pool"],...
    ["global_average_pooling2d_1"],...
    ["pool5"],...
    ["avg_pool"],...
    ["pool5"],...
    ["drop7"]
    };
layerArraySave = {["new_fc"],...
    ["new_fc"],...
    ["new_fc"],...
    ["new_fc"],...
    ["new_fc"],...
    ["new_fc"],...
    ["fc7"]
    };

nameArray=["googleNet","inceptionv3","mobilenetv2","restnet18","restnet50","restnet101","vgg19"];


fileArray=["googlenet__EP20__MBS32.mat",...
    "inceptionv3__EP20__MBS32.mat",...
    "mobilenetv2__EP20__MBS32.mat",...
    "resnet18__EP20__MBS32.mat",...
    "resnet50__EP20__MBS32.mat",...
    "resnet101__EP20__MBS32.mat",...
    "vgg19__EP20__MBS32.mat"
    ];
n = 10;
for f=8:length(fileArray)
trainTable=load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\train\TOT_ALL_train_table.mat').trainTable;
testingTable=load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test\TOT_ALL_test_table.mat').testingTable;
    %dataset
imdsTrain=imageDatastore('D:\Tesi\DataIncRGB\train','IncludeSubFolders',true','LabelSource','foldernames');
imdsTest=imageDatastore('D:\Tesi\DataIncRGB\test','IncludeSubFolders',true','LabelSource','foldernames');


%imdsTrain=splitEachLabel(imdsTrain,0.001);
%imdsTest=splitEachLabel(imdsTest,0.001);
%trainTable=head(trainTable,143);
%testingTable=head(testingTable,25);


if f==1
    imdsTrain.ReadFcn = @customReadDatastoreImage224;
    imdsTest.ReadFcn = @customReadDatastoreImage224;
elseif f==2
    imdsTrain.ReadFcn = @customReadDatastoreImage299;
    imdsTest.ReadFcn = @customReadDatastoreImage299;
else
    imdsTrain.ReadFcn = @customReadDatastoreImage224;
    imdsTest.ReadFcn = @customReadDatastoreImage224;
end

    
    
    
    
    
    trainedNet=load(fullfile(rootPath, classifiersPath, fileArray(f)))
    fprintf('%s%d',"load del file ",f);
    
    
  
    
    
    %newTrain=table(trainTable.labels,horzcat(trainTable.features,featuresTrain));
    
 
        tic;
        featuresTrain =[];
        featuresTrainVert =[];
        fprintf('%s%d%s%d%s%d\n', "Partizione Train: ",i," su ",n," numero files ",length(imdsTrain.Files));
        filename=strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Train\",nameArray(f),"_Plus_ALL_Traditional_Features_Train");
        for j=1:length(layerArray{f})
            layers=layerArray{f};
            featuresTrainInner = activations(trainedNet.trainedNet,imdsTrain,layers(j),'OutputAs','rows', 'MiniBatchSize', 32);
            featuresTrainInner = squeeze(featuresTrainInner);
            featuresTrainVert=horzcat(featuresTrainVert,featuresTrainInner);
            disp(size(featuresTrainVert));
            disp(size(featuresTrainInner));
        end
        
        trainTable=table(trainTable.labels,featuresTrainVert);
        save(filename,'trainTable','-v7.3');
        toc;
  
    
    
        tic;
        featuresTest =[];
        featuresTestVert =[];
        filename=strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Test\",nameArray(f),"_Plus_ALL_Traditional_Features_Test");
        fprintf('%s%d%s%d%s%d\n', "Partizione Test: ",i," su ",n," numero files ",length(imdsTest.Files));
        for j=1:length(layerArray{f})
            layers=layerArray{f};
            featuresTestInner = activations(trainedNet.trainedNet,imdsTest,layers(j),'OutputAs','rows', 'MiniBatchSize', 32);
            featuresTestInner = squeeze(featuresTestInner);
            featuresTestVert=horzcat(featuresTestVert,featuresTestInner);
        end
         
         testingTable=table(testingTable.labels,featuresTestVert);
        save(filename,'testingTable','-v7.3');
        toc;
end

%fprintf('%s%d%d\n', "File totali: ",length(imdsTrain.Files));
%fprintf('%s%d%d\n', "File totali: ",length(imdsTest.Files));

%newTest=table(testingTable.labels,horzcat(testingTable.features,featuresTest));


function data = customReadDatastoreImage224(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = imresize(data,[224 224]);
end

function data = customReadDatastoreImage299(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = imresize(data,[299 299]);
end
end

