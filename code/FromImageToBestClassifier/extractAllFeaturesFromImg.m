function [label,features,time] = extractAllFeaturesFromImg(img,imgRGB,info,arrayCNN)
%EXTRACTALLFEATURES Summary of this function goes here
%   Detailed explanation goes here
timeinit=tic;
arrayDescriptors = {'HM',...
    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6'...
    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
    'HARri','LBP18'};
color = 'gray';
graylevel = 256;
prepro = 'none';
%calcola le features hand-crafted
featuresHC=[];

if not(size(img,1)==512 && size(img,2)==512)
         imgHC=imresize(img,[512,512]);
else
    imgHC=img;
    
end
label=info;
for indexFeature=1:length(arrayDescriptors)
    featuresHC=horzcat(featuresHC,featureExtraction(imgHC, arrayDescriptors{indexFeature}, color, graylevel, prepro).');
end
%calcola features cnn 

layerArray = [...
    "fc7",...
    "pool5-drop_7x7_s1",...
    "avg_pool",...
    "global_average_pooling2d_1",...
    "pool5",...
    "pool5",...
    "avg_pool",...
    "node_200",...
    "drop7",...
    "drop7"];

%myFiles = dir(fullfile(cnnFolder,'*.mat')); %gets all wav files in struct
features={length(arrayCNN)};
for indexCNN = 1:length(arrayCNN)
    %resize a seconda della cnn
    if indexCNN == 1
         imgCNN=imresize(imgRGB,[227,227]);
    elseif indexCNN == 3
         imgCNN=imresize(imgRGB,[299,299]);
    else
         imgCNN=imresize(imgRGB,[224,224]);
    end
    %estrae e concatena con quelle hc
    A_cell = struct2cell(arrayCNN{indexCNN});
    D= A_cell{1};
    features{indexCNN} = horzcat(activations(D,imgCNN,layerArray(indexCNN),'OutputAs','rows'),featuresHC);
end

time=toc(timeinit);
end