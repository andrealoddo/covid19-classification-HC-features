% Esempio di calcolo delle feature per una singola immagine
addpath(genpath('featuresComputation'));
addpath(genpath('HAR'));
img = imread('../img.png');

descriptor = 'LBP18'; % 'HAR', o 'LBP18'
color = 'gray';
graylevel = 256;
prepro = 'none';


descriptors_sets = {'HM',...
                    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6','ZM_10_8','ZM_10_10',...
                    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
                    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
                    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
                    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
                    'HARri','LBP18'};    
                

features = featureExtraction(img, descriptor, color, graylevel, prepro);

%Esempio di calcolo features per piu' immagini:
%img = datastore(....) o funzioni simili

%for i = 1:size(img)
    
    %features = [features; featureExtraction(img, descriptor, color, graylevel, prepro)'];
    
%end


%preparo il dataset per il training
%[trainHAR,validateHAR]=extractFeaturesAndSplit('HAR',0.8);
%[trainLBP18,validateLBP18]=extractFeaturesAndSplit('LBP18',0.8);
[trainLBP18e,validateLBP18e,testLBP18e]=concatenateFeatures();
%[trainLBP18e,validateLBP18e,testLBP18e]=ettExtractFeaturesAndSplit('LBP18');
%[trainHARe,validateHARe,testHARe]=ettExtractFeaturesAndSplit('HAR');
%eseguo il predict tramite il modello 
%essendo il dataset ancora di piccole dimensioni i risultati possono essere
%falsati, aumentando il fold è possibile ottenere risultati più concreti
%prima di eseguire questa porzione è necessario avere un modello già
%allenato
    predictions = trainedModel.predictFcn(testLBP18e);
%calcolo l'accuratezza

    iscorrect=predictions==testLBP18e.labels;
    iscorrect=iscorrect(:,1);
    sizeArray=size(trainLBP18e);
    accuracy=sum(iscorrect)*100/length(testLBP18e.labels);

