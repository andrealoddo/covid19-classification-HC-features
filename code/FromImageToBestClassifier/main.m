imgPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset";
classifersPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedClassifiers";
cnnPath="D:\Tesi\covid19-classification-HC-features\code\PretrainedCNN";
featuresPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedFeatures";
outputTextPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedStats";

%carico tutte le cnn
myFiles = dir(fullfile(cnnPath,'*.mat')); %gets all wav files in struct
arrayCNN={length(myFiles)};
for cnnIndex=1:length(myFiles)
    arrayCNN{cnnIndex}=load(myFiles(cnnIndex).name);
end
%estraggo le features per tutto il dataset

featuresCNN={length(myFiles)};
labels=[];
times=[];
index=1;
nameArray=["features_alexnetHC","features_googleNetHC","features_inceptionv3HC","features_mobilenetv2HC","features_restnet101HC",...
    "features_restnet18HC","features_restnet50HC","features_shufflenetHC","features_vgg16HC","features_vgg19HC"];
nameArrayClassifiers=["trained_boosted_trees_alexnetHC","trained_boosted_trees_googleNetHC","trained_boosted_trees_inceptionv3HC","trained_boosted_trees_mobilenetv2HC","trained_boosted_trees_restnet101HC",...
    "trained_boosted_trees_restnet18HC","trained_boosted_trees_restnet50HC","trained_boosted_trees_shufflenetHC","trained_boosted_trees_vgg16HC","trained_boosted_trees_vgg19HC"];
fprintf('%s\n', "Estrae train ");
imds_train=imageDatastore(strcat(imgPath,"\train"),'IncludeSubFolders',true','LabelSource','foldernames');
while(hasdata(imds_train))
    fprintf('\n%s%d', "Estrazione features immagine ",index);
    [grayImage,info]=read(imds_train);
    if size(grayImage,3)==3
        grayImage=rgb2gray(grayImage);
    end
    filler = zeros(size(grayImage),'uint8');
    rgbImage = cat(3, grayImage, grayImage, grayImage);
    [labelImg,featuresImg,time]=extractAllFeaturesFromImg(grayImage,rgbImage,info,arrayCNN);
    for cnnIndex=1:length(myFiles)
        if index==1
            featuresCNN{cnnIndex}=featuresImg{cnnIndex};
        else
            featuresCNN{cnnIndex}=vertcat(featuresCNN{cnnIndex},featuresImg{cnnIndex});
        end
    end
    index=index+1;
    labels=vertcat(labels,labelImg.Label);
    fprintf('%s%d', "Last time: ",time);
end
%salvataggio features
for cnnIndex=1:length(myFiles)
    outputTable=table(labels,featuresCNN{cnnIndex});
    save(strcat(featuresPath,"\train\",nameArray(cnnIndex)),'outputTable','-v7.3');
end

fprintf('%s\n', "Estrae test ");
imds_test=imageDatastore(strcat(imgPath,"\test"),'IncludeSubFolders',true','LabelSource','foldernames');
while(hasdata(imds_test))
    fprintf('\n%s%d', "Estrazione features immagine ",index);
    [grayImage,info]=read(imds_test);
    if size(grayImage,3)==3
        grayImage=rgb2gray(grayImage);
    end
    filler = zeros(size(grayImage),'uint8');
    rgbImage = cat(3, grayImage, grayImage, grayImage);
    [labelImg,featuresImg,time]=extractAllFeaturesFromImg(grayImage,rgbImage,info,arrayCNN);
    for cnnIndex=1:length(myFiles)
        if index==1
            featuresCNN{cnnIndex}=featuresImg{cnnIndex};
        else
            featuresCNN{cnnIndex}=vertcat(featuresCNN{cnnIndex},featuresImg{cnnIndex});
        end
    end
    index=index+1;
    labels=vertcat(labels,labelImg.Label);
    fprintf('%s%d', "Last time: ",time);
end
%salvataggio features
for cnnIndex=1:length(myFiles)
    outputTable=table(labels,featuresCNN{cnnIndex});
    save(strcat(featuresPath,"\test\",nameArray(cnnIndex)),'outputTable','-v7.3');
end


fprintf('%s\n', "Addestra classificatori ");
%zona addestramento
myDirFeaturesTrain = 'D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedFeatures\train'; %gets directory
myFilesFeaturesTrain = dir(fullfile(myDirFeaturesTrain,'\*.mat')); %gets all wav files in struct

for k = 1:length(myFilesFeaturesTrain)
    fprintf('%s%d\n', "Addestra classificatori ",k);
    baseFileNameFeaturesTrain = myFilesFeaturesTrain(k).name;
    fullFileNameFeaturesTrain = fullfile(baseFileNameFeaturesTrain);
    trainTable=load(fullFileNameFeaturesTrain).outputTable;
    %trainTable=head(trainTable,180);
    trainTable.Properties.VariableNames{'Var2'} = 'features';
   
    
    timeinit=tic;
    [trainedClassifier,~]=ensembleBoostedTreesNew(trainTable);
    trainedClassifier.classifierType="ensembleBoostedTrees";
    classifier=trainedClassifier;
    timefin=toc(timeinit);
    classifier.trainingTimeSeconds=timefin;
    save(strcat(classifersPath,"\", nameArrayClassifiers(k)), 'classifier','-v7.3');
end


fprintf('%s\n', "ensembla ");
%myDir = 'D:\Tesi\covid19-classification-HC-features\code\trainedClassifiers'; %gets directory
%fixTables();
myDir = classifersPath; %gets directory
myFiles = dir(fullfile(myDir,'*.mat')); %gets all wav files in struct
myDirFeaturesTest = 'D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedFeatures\test'; %gets directory
myFilesFeaturesTest = dir(fullfile(myDirFeaturesTest,'*.mat')); %gets all wav files in struct
predictions=[];
originLabels=[];
for k = 1:length(myFiles)
    
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(baseFileName);
    baseFileNameFeaturesTest = myFilesFeaturesTest(k).name;
    fullFileNameFeaturesTest = fullfile(baseFileNameFeaturesTest);
    testingTable=load(fullFileNameFeaturesTest).outputTable;
    testingTable.Properties.VariableNames{'Var2'} = 'features';
    fprintf('%s%s\n', "Inizio ",fullFileName);
    load(fullFileName);
    predictionSingle=classifier.predictFcn(testingTable);
    predictions=horzcat(predictions,predictionSingle);
    originLabels=testingTable.labels;
    
end
fprintf('%s\n', "vota ");
votes=mode(predictions,2);

%crea matrice confusione e statistiche
confusion = confusionmat(originLabels,votes);
[stats, texMacro, texMicro] = computeStats(confusion);
%modifico la table
mA=table2array(modifiedTable);
fileID = fopen(strcat("marioDraghi.txt"),'2');
fprintf(fileID,['\\begin{table}[htbp]\n',...
    '\\centering\n',...
    '\\begin{tabular}{|p{2.9cm}||p{2cm}|p{2cm}|p{2cm}|p{2.5cm}|p{2.5cm}|}\n',...
    '\\hline\n',...
    'Nome & COVID-19 & Normale & Polmonite & Macro Avg & Micro Avg\\\\\n',...
    '\\hline\n',...
    'True Positive & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'False Positive & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'False Negative & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'True Negative & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'Precision & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'Sensitivity & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'Spefificaticy & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'Accuracy & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'F-measure & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'MAvG & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    'MAvA & %s & %s & %s & %s & %s\\\\\n',...
    '\\hline\n',...
    '\\end{tabular}\n',...
    '\\caption{}\n',...
    '\\label{tab:}\n',...
    '\\end{table}\n\\\\\\\\'
    ],mA(1,2),mA(1,3),mA(1,4),mA(1,5),mA(1,6)...Chiedo venia per lo stampino
    ,mA(2,2),mA(2,3),mA(2,4),mA(2,5),mA(2,6)...
    ,mA(3,2),mA(3,3),mA(3,4),mA(3,5),mA(3,6)...
    ,mA(4,2),mA(4,3),mA(4,4),mA(4,5),mA(4,6)...
    ,mA(5,2),mA(5,3),mA(5,4),mA(5,5),mA(5,6)...
    ,mA(6,2),mA(6,3),mA(6,4),mA(6,5),mA(6,6)...
    ,mA(7,2),mA(7,3),mA(7,4),mA(7,5),mA(7,6)...
    ,mA(8,2),mA(8,3),mA(8,4),mA(8,5),mA(8,6)...
    ,mA(9,2),mA(9,3),mA(9,4),mA(9,5),mA(9,6)...
    ,mA(10,2),mA(10,3),mA(10,4),mA(10,5),mA(10,6)...
    ,mA(11,2),mA(11,3),mA(11,4),mA(11,5),mA(11,6)...
    );
fclose(fileID);
fprintf('%s\n', "Fine ");




