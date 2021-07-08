featuresPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\ComputedFeatures";
myFiles = dir(fullfile(featuresPath,'*.mat')); %gets all wav files in struct
nameArray=["features_alexnetHC","features_googleNetHC","features_inceptionv3HC","features_mobilenetv2HC","features_restnet101HC",...
    "features_restnet18HC","features_restnet50HC","features_shufflenetHC","features_vgg16HC","features_vgg19HC"];
imgPath="D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset";
imds=imageDatastore(imgPath,'IncludeSubFolders',true','LabelSource','foldernames');
labels=[];
while(hasdata(imds))
    [grayImage,info]=read(imds);
    labels=vertcat(labels,info.Label);
end
for i=1:length(myFiles)
    load(myFiles(i).name)
    outputTable=table(labels,outputTable.Var2);
    save(strcat(featuresPath,"\",nameArray(i)),'outputTable','-v7.3');
end