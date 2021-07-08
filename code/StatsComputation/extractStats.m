classifiersPath="D:\Unica\Università di Cagliari\Andrea Loddo - covid_classification_models";
featuresPath="D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test";
outputPath="D:\Tesi\covid19-classification-HC-features\code\StatsComputation\ComputedStats";
arrayDescriptors = {'HM',...
    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6'...
    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
    'HARri','LBP18'};
%tutti i file utili
myClassifiers = dir(fullfile(classifiersPath,'**/*.mat'));
myFeatures = dir(fullfile(featuresPath,'**/*.mat'));
%estraggo i nomi per i classificatori
classifiersFolders={};
for i=1:length(myClassifiers)
    [~, ParentFolderName] = fileparts(myClassifiers(i).folder) ;
    classifiersFolders=vertcat(classifiersFolders,ParentFolderName);
end
classifiersFolders=unique(classifiersFolders);
for i=1:length(classifiersFolders)
    mkdir(strcat(outputPath,"\",classifiersFolders{i}));
end
%opero estraendo le statistiche

for c=1:length(classifiersFolders)
    for f=1:length(arrayDescriptors)
        destinationName=strcat(outputPath,"/",classifiersFolders{c},"/",classifiersFolders{c},"_",arrayDescriptors(f),"_report");
        classifierName=strcat(classifiersPath,"\",classifiersFolders{c},"\","trained_",classifiersFolders{c},"_",arrayDescriptors(f),".mat");
        if( exist( fullfile( classifierName) , 'file') == 2 && not(exist( fullfile(strcat( destinationName,".mat")) , 'file') == 2 ))
            fprintf('%s%s\n',classifierName ," esiste");
            testingTableName=strcat(arrayDescriptors(f),"_test_table.mat");
            load(testingTableName);
            
            load(classifierName);
            predictions=classifier.predictFcn(testingTable);
            confusion = confusionmat(testingTable.labels,predictions);
            [stats, texMacro, texMicro] = computeStats(confusion);
            %modifico la table
            modifiedTable = splitvars(stats,'classes','NewVariableNames',{'covid-19','normal','pneumonia'});
            toSave=table2array(modifiedTable);
            
            save(destinationName, 'toSave','-v7.3');
        else
            fprintf('%s%s\n',classifierName ," non esiste");
        end
    end
end




timetot=0;
for i=1:length(myClassifiers)
    load(myClassifiers(i).name);
    timetot=timetot+classifier.trainingTimeSeconds;
end









