function [out] = utilityClassifiers(thing,topElements)
%UTILITYCLASSIFIERS Summary of this function goes here
%   Detailed explanation goes here

if nargin==2
    if thing=="time"
        out=getAllTimes();
        out=getTop(out,'ascend',topElements);
    end
    if thing=="accuracy"
        out=getAllAccuracy();
        out=getTop(out,'descend',topElements);
    end
else
     if thing=="time"
        out=getAllTimes();
        out=getTop(out,'ascend');
    end
    if thing=="accuracy"
        out=getAllAccuracy();
        out=getTop(out,'descend');
    end
    if thing=="vote"
        [votes,accuracy,usedClassifiers]=getBestVotes();
        out=accuracy;
    end
end
end
%generazione classificatore tramite votazione votazione
function out=combineClassifiers(arrayClassifiersNames)
    array={};
    for i=1:length(arrayClassifiersNames)
        load( arrayClassifiersNames(i,1));
        array=vertcat(array,classifier);
        
            %manda fuori errore
        
    end
    out=array;
end
function [votes,accuracy,usedClassifiers]=evalVotes(arrayClassifiers)
    arrayPredictions=[];
    for i=1:length(arrayClassifiers)
        fprintf('%s%d\n', "Computo per elemnto_ ",i);
            load(strcat("D:\Tesi\FeaturesSingleCpu\test\",arrayClassifiers{i,1}.usedFeature,"_test_table"));
            fprintf('%s%s\n', "Caricato:  ",strcat("D:\Tesi\FeaturesSingleCpu\test\",arrayClassifiers{i,1}.usedFeature,"_test_table"));
            arrayPredictions=horzcat(arrayPredictions,arrayClassifiers{i}.predictFcn(testingTable));
            fprintf('%s\n', "Messo nell'array ");
            trueLabels=testingTable.labels;
        
    end
     fprintf('%s\n', "Predizioni");
    disp(arrayPredictions);
    votes=mode(arrayPredictions,2);
    %uso una a caso tanto le labels son tutte uguali
    accuracy=0.;
    for i=1:length(votes)
        if(trueLabels(i,1)==votes(i,1))
             accuracy=accuracy+1;
        end
    end
    accuracy=accuracy/length(votes);   
    usedClassifiers=arrayClassifiers;
end
function [votes,accuracy,usedClassifiers]=getBestVotes()
    [votes,accuracy,usedClassifiers]=evalVotes(combineClassifiers(getTop(getAllAccuracy(),'descend',60)));
    %[votes,accuracy,usedClassifiers]=evalVotes(combineClassifiers(getAllAccuracy()));
end
%generazione info dei classificatori 
function out=getAllTimes()
    descriptors_sets = {'HM',...
                    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6',...
                    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
                    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
                    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
                    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
                    'HARri','LBP18'}; 
                
    classifiersNames = {'coarseGaussianSvm','ensembleBeggedTrees','fineTree','gaussianNaiveBayes',...
        'kernelNaiveBayes','linearSvm','mediumGaussianSvm','mediumNeuralNetwork','narrowNeuralNetwork',...
        'quadraticSvm','rusBoostedTrees','weightedKnn'}; 
    timeArray=[];
    nameArray=[];
    for x=1:length(descriptors_sets)
        for y=1:length(classifiersNames)
            if not(exist(strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x},".mat"))==0)
         
             load(strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x}));
             timeArray=vertcat(timeArray,classifier.trainingTimeSeconds);
             nameArray=vertcat(nameArray,strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x}));
            end
        end
    end   
    out=horzcat(nameArray,timeArray);
end
function out=getAllAccuracy()
    descriptors_sets = {'HM',...
                    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6',...
                    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
                    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
                    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
                    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
                    'HARri','LBP18'}; 
                
    classifiersNames = {'coarseGaussianSvm','ensembleBeggedTrees','fineTree','gaussianNaiveBayes',...
        'kernelNaiveBayes','linearSvm','mediumGaussianSvm','mediumNeuralNetwork','narrowNeuralNetwork',...
        'quadraticSvm','rusBoostedTrees','weightedKnn'}; 
    accuracyArray=[];
    nameArray=[];
    for x=1:length(descriptors_sets)
        for y=1:length(classifiersNames)

            if not(exist(strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x},".mat"))==0)
         
             load(strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x}));
             accuracyArray=vertcat(accuracyArray,classifier.testedAccuracy);
             nameArray=vertcat(nameArray,strcat("trainedClassifiers/",classifiersNames{y},"/trained_",...
             classifiersNames{y},"_", descriptors_sets{x}));
            end
        end
    end   
    out=horzcat(nameArray,accuracyArray);
end
function out=getTop(in,mode,n)
    valuesIn=in(:,2);
    namesIn=in(:,1);
    if mode=="ascend"
        [valuesIn,isort]=sort(valuesIn);
         fprintf('%s\n', "ascend ");
    else
        [valuesIn,isort]=sort(valuesIn,mode);
         fprintf('%s\n', "desc ");
    end
    namesIn=namesIn(isort);
    out=horzcat(namesIn,valuesIn);
    if nargin==3
        out=out(1:n,:);
    end
end
