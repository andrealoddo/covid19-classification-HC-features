%move images
addpath( 'C:/Users/loand/Google Drive/Ricerca/Codes/MATLAB/Utilities' )

% labels: 0 Normal, 1 Pneumonia, 2 COVID-19
mainDir = '2A_images';
images = dir( fullfile('.', mainDir, '*.png') );
images = {images.name}.';
images = sort_nat(images);
%metadata = readtable('metadata.csv', 'Delimiter',',');

trainSet = readtable('train_COVIDx_CT-2A.txt', 'Delimiter',' ', 'PreserveVariableNames', true);
trainSet.Properties.VariableNames{1} = 'images';
trainSet.Properties.VariableNames{2} = 'labels';

validLabels = unique(trainSet.labels);

if( ~exist('train', 'dir') )
    mkdir('train');
    if( ~exist('train/normal', 'dir') )
        mkdir('train/normal');
    end
    if( ~exist('train/pneumonia', 'dir') )
        mkdir('train/pneumonia');
    end
    if( ~exist('train/covid-19', 'dir') )
        mkdir('train/covid-19');
    end
    
    for i = 1:numel(trainSet.images)
    
        label = trainSet.labels(i);

        if(label == 0) % normal
            copyfile( fullfile(mainDir, trainSet.images{i}), 'train/normal');
        elseif(label == 1) % pneumonia
            copyfile( fullfile(mainDir, trainSet.images{i}), 'train/pneumonia');
        elseif(label == 2) % covid-19
            copyfile( fullfile(mainDir, trainSet.images{i}), 'train/covid-19');
        end
    
    end

end

