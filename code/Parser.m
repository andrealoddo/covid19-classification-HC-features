%move images
addpath( 'D:\Tesi\DataInc' )

% labels: 0 Normal, 1 Pneumonia, 2 COVID-19
mainDir = '2A_images';
images = dir( fullfile('D:/Tesi/DataInc', mainDir, '*.png') );
images = {images.name}.';
images = sort_nat(images);
%metadata = readtable('metadata.csv', 'Delimiter',',');

validateSet = readtable('val_COVIDx_CT-2A.txt', 'Delimiter',' ', 'PreserveVariableNames', true);
validateSet.Properties.VariableNames{1} = 'images';
validateSet.Properties.VariableNames{2} = 'labels';

validLabels = unique(validateSet.labels);
 fprintf('%s\n', "?????  ");
if( ~exist('D:\Tesi\DataInc\validate', 'dir') )
    mkdir('D:\Tesi\DataInc\validate');
    if( ~exist('D:\Tesi\DataInc\validate/normal', 'dir') )
        mkdir('D:\Tesi\DataInc\validate/normal');
    end
    if( ~exist('D:\Tesi\DataInc\validate/pneumonia', 'dir') )
        mkdir('D:\Tesi\DataInc\validate/pneumonia');
    end
    if( ~exist('D:\Tesi\DataInc\validate/covid-19', 'dir') )
        mkdir('D:\Tesi\DataInc\validate/covid-19');
    end
    fprintf('%s\n', "?????  ");
    for i = 1:numel(validateSet.images)
        label = validateSet.labels(i);

        if(label == 0) % normal
            copyfile( fullfile('D:\Tesi\DataInc\2A_images', validateSet.images{i}), 'D:\Tesi\DataInc\validate/normal');
        elseif(label == 1) % pneumonia
            copyfile( fullfile('D:\Tesi\DataInc\2A_images', validateSet.images{i}), 'D:\Tesi\DataInc\validate/pneumonia');
        elseif(label == 2) % covid-19
            copyfile( fullfile('D:\Tesi\DataInc\2A_images', validateSet.images{i}), 'D:\Tesi\DataInc\validate/covid-19');
        end
    
    end

end

