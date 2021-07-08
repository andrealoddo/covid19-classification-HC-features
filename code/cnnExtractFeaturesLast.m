function [] = cnnExtractFeaturesLast()
if ispc
    % Windows dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    classifiersPath = '\PretrainedCNN\';
   
    % WS dataset path
    rootPath = 'D:\Tesi\covid19-classification-HC-features\code';
    classifiersPath = '\PretrainedCNN\';
end
%ogni volta ricarico per certezza che non vengano consumati



layerArray = {["fc7"],...
    ["node_200"],...
    ["drop7"]
    };


nameArray=["alexnet","shufflenet","vgg16"];


fileArray=["alexnet__EP20__MBS32",...
    "shufflenet__EP20__MBS32.mat",...
    "vgg16__EP20__MBS32.mat"];
n = 10;
%for f=3:length(fileArray)
for f=1:1

    %dataset
trainTable=load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\train\TOT_ALL_train_table.mat').trainTable;
testingTable=load('D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test\TOT_ALL_test_table.mat').testingTable;
imdsTrain=imageDatastore('D:\Tesi\DataIncRGB\train','IncludeSubFolders',true','LabelSource','foldernames');
imdsTest=imageDatastore('D:\Tesi\DataIncRGB\test','IncludeSubFolders',true','LabelSource','foldernames');


%imdsTrain=splitEachLabel(imdsTrain,0.001);
%imdsTest=splitEachLabel(imdsTest,0.001);
%trainTable=head(trainTable,143);
%testingTable=head(testingTable,25);

if f==1
    imdsTrain.ReadFcn = @customReadDatastoreImage227;
    imdsTest.ReadFcn = @customReadDatastoreImage227;
elseif f==2
    imdsTrain.ReadFcn = @customReadDatastoreImage224;
    imdsTest.ReadFcn = @customReadDatastoreImage224;
else
    imdsTrain.ReadFcn = @customReadDatastoreImage224;
    imdsTest.ReadFcn = @customReadDatastoreImage224;
end

    
    
    
    
    
    trainedNet=load(fullfile(rootPath, classifiersPath, fileArray(f)));
    fprintf('%s%d',"load del file ",f);
    
    
  
    
    
    %newTrain=table(trainTable.labels,horzcat(trainTable.features,featuresTrain));
    
 
        tic;
        featuresTrain =[];
        featuresTrainVert =[];
        fprintf('%s%d%s%d%s%d\n', "Partizione Train: ",i," su ",n," numero files ",length(imdsTrain.Files));
        filename=strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Train\",nameArray(f),"v2_Plus_ALL_Traditional_Features_Train");
        for j=1:length(layerArray{f})
            layers=layerArray{f};
            A_cell = struct2cell(trainedNet);
            D= A_cell{1};
            featuresTrainInner = activations(D,imdsTrain,layers(j),'OutputAs','rows', 'MiniBatchSize', 32);
            featuresTrainVert=horzcat(featuresTrainVert,featuresTrainInner);
        end
         trainTable=table(trainTable.labels,featuresTrainVert);
        save(filename,'trainTable','-v7.3');
        toc;
  
    
    
        tic;
        featuresTest =[];
        featuresTestVert =[];
        filename=strcat("D:\Tesi\covid19-classification-HC-features\code\FeaturesCnn\Test\",nameArray(f),"v2_Plus_ALL_Traditional_Features_Test");
        fprintf('%s%d%s%d%s%d\n', "Partizione Test: ",i," su ",n," numero files ",length(imdsTest.Files));
        for j=1:length(layerArray{f})
            layers=layerArray{f};
             A_cell = struct2cell(trainedNet);
            D= A_cell{1};
            featuresTestInner = activations(D,imdsTest,layers(j),'OutputAs','rows', 'MiniBatchSize', 32);
            featuresTestVert=horzcat(featuresTestVert,featuresTestInner);
        end
        testingTable=table(testingTable.labels,featuresTestVert);
        save(filename,'testingTable','-v7.3');
        toc;
end

fprintf('%s%d%d\n', "File totali: ",length(imdsTrain.Files));
fprintf('%s%d%d\n', "File totali: ",length(imdsTest.Files));

%newTest=table(testingTable.labels,horzcat(testingTable.features,featuresTest));


function data = customReadDatastoreImage227(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = imresize(data,[227 227]);
end

function data = customReadDatastoreImage299(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = imresize(data,[299 299]);
end
function data = customReadDatastoreImage224(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = imresize(data,[224 224]);
end

end

