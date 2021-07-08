classifiersPath="D:\Unica\Università di Cagliari\Andrea Loddo - covid_classification_models";
featuresPath="D:\Tesi\covid19-classification-HC-features\code\FeaturesSingleCpu\test";
statsPath="D:\Tesi\covid19-classification-HC-features\code\StatsComputation\ComputedStats";
arrayDescriptors = {'HM',...
    'ZM_4_2','ZM_4_4','ZM_5_3','ZM_5_5','ZM_6_2','ZM_6_4','ZM_6_6','ZM_7_3','ZM_7_5','ZM_7_7','ZM_7_8','ZM_7_9','ZM_8_2','ZM_8_4','ZM_8_6','ZM_8_8','ZM_9_3','ZM_9_5','ZM_9_7','ZM_9_9','ZM_9_11','ZM_10_2','ZM_10_4','ZM_10_6'...
    'LMG_3','LMG_4','LMG_5','LMG_6','LMG_7','LMG_8','LMG_9','LMG_10',...
    'LMGS_3','LMGS_4','LMGS_5','LMGS_6','LMGS_7','LMGS_8','LMGS_9','LMGS_10',...
    'CH_3','CH_4','CH_5','CH_6','CH_7','CH_8','CH_9','CH_10',...
    'CHdue_3','CHdue_4','CHdue_5','CHdue_6','CHdue_7','CHdue_8','CHdue_9','CHdue_10',...
    'HARri','LBP18'};
utils=["ZM_","LMG_","LMGS_","CH_","CHdue_","HARri","HM","LBP18"];
%tutti i file utili
for h=1:length(utils)
myClassifiers = dir(fullfile(classifiersPath,'**/*.mat'));
myFeatures = dir(fullfile(featuresPath,'**/*.mat'));
%estraggo i nomi per i classificatori
classifiersFolders={};
for i=1:length(myClassifiers)
    [~, ParentFolderName] = fileparts(myClassifiers(i).folder) ;
    classifiersFolders=vertcat(classifiersFolders,ParentFolderName);
end
classifiersFolders=unique(classifiersFolders);


%da ora ricerco
results={};
myStats = dir(fullfile(statsPath,'**/*.mat'));
pattern=utils(h);%pattern del momento da ricercare
%pattern=["HARri","HM","LBP18"];
class=4;%2 covid, 3 normal, 4 pneumonia
stat=5;%5 precision, 6 sensitivity(recall), 7 specificity, 8 accuracy, 9 f-measure

%trovo tutti quelli con la feature desiderata
for i=1:length(myStats)
    if contains(myStats(i).name,pattern)
        results{i}=myStats(i).name;
    end
end
%tolgo celle vuote
results=results(~cellfun('isempty',results));
results=sort_nat(results);
%prendo i nomi delle features
features={};
for i=1:length(classifiersFolders)
    features=erase(results,strcat(classifiersFolders,"_"));
    features=erase(features,"_report.mat");
end
features=unique(features);
features=sort_nat(features);
output={};
%posizioni
for i=1:length(features)
    for j=1:length(classifiersFolders)
        filteredF=contains(results,features{i});
        filteredC=contains(results,classifiersFolders{j});
        filtered=filteredF&filteredC;
        pos=find(filtered);
        if not( isempty(pos))
            fprintf('%s\n', "?????  ");
            stats=load(results{pos}).toSave;
            m=stats(stat,class);
            output{i+1,j+1}=m;
            output{1,j+1}=classifiersFolders{j};
        else
             output{i+1,j+1}="";
        end
    end
    output{i+1,1}=features{i};
end
output{1,1}="-";
output=string(output);
pescoso=str2double(output(2:length(features)+1,2:length(classifiersFolders)+1));
hhh{3,h}=max(max(pescoso));
end
output=string(output);
bar3(str2double(output(2:length(features)+1,2:length(classifiersFolders)+1)));
ax = gca;
yticks(1:length(features));
labelsy=string(features);
labelsy=strrep(labelsy,"_","\_");
set(gca,'YTickLabel',labelsy);
labelsx=string(classifiersFolders);
labelsx=strrep(labelsx,"_","\_");
set(gca, 'XTickLabel', labelsx);
ax.FontSize = 14;


















