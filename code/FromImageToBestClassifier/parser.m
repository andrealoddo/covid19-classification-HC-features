%move images


% labels: 0 Normal, 1 Pneumonia, 2 COVID-19
mainDir = 'D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\normal';
images = dir( fullfile( mainDir, '*.png') );
images = {images.name}.';
images = sort_nat(images);
%metadata = readtable('metadata.csv', 'Delimiter',',');

validateSet = readtable('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\valCT_NonCOVID.txt','Delimiter', ',');

 fprintf('%s\n', "?????  ");
if( ~exist('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate', 'dir') )
    mkdir('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate');
    if( ~exist('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/normal', 'dir') )
        mkdir('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/normal');
    end
    if( ~exist('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/covid-19', 'dir') )
        mkdir('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/covid-19');
    end
    fprintf('%s\n', "?????  ");
   

end
set=table2array(validateSet);
for i = 1:size(set)
        img = set(i);
        img=img{1};

            filename=strcat('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\normal\',img);
            %copyfile( fullfile('D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\normal\', img), 'D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/normal');
            copyfile( filename, 'D:\Tesi\covid19-classification-HC-features\code\FromImageToBestClassifier\dataset\validate/normal');

    
end
