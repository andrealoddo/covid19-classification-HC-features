                                models = dir( fullfile( modelsPath, string(datasetsname(dt)), aug, string(folders_split(fs)), '*.mat' ) );
                                if contains(descriptors{dsc},'FTalex') %%%alexnet CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'alex' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTVGG16') %%%vgg16 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'vgg16' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTVGG19') %%%vgg19 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'vgg19' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTgoogle') %%%google CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'google' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTresnet50') %%%resnet50 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'resnet50' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTresnet18') %%%resnet18 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'resnet18' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTresnet101') %%%resnet101 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'resnet101' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                elseif contains(descriptors{dsc},'FTInceptionv3') %%%inceptionv3 CNN Features
                                    convnet_ind = find( contains( {models.name}.', 'inceptionv3' ) );
                                    load( fullfile(models(convnet_ind).folder, models(convnet_ind).name ) );
                                    disp( strcat('loaded ', models(convnet_ind).name, ' neural network') )
                                end
                                convnet = trainedNet;


                                    if contains(descriptors{dsc},'CNN') %%%CNN Features
                                        imds.ReadFcn = @(filename)readAndPreprocessImage(filename,  prepro{pp}, [sizes(1) sizes(2)], graylevel(gl));
                                        if contains(descriptors{dsc},'alexfc7') %%%alexnet Features                                                            
                                            features = activations(convnet, imds, 'fc7', 'MiniBatchSize', 32);
                                        elseif contains(descriptors{dsc},'VGG') %%%VGG CNN Features
                                            features = activations(convnet, imds, 'fc7', 'MiniBatchSize', 32);
                                        elseif contains(descriptors{dsc},'google') %%%CNN Features
                                            features = activations(convnet, imds, 'new_fc', 'MiniBatchSize', 32);
                                        elseif contains(descriptors{dsc},'resnet') %%%CNN Features
                                            features = activations(convnet, imds, 'new_fc', 'MiniBatchSize', 32);
                                        elseif contains(descriptors{dsc},'Inception') %%%CNN Features
                                            features = activations(convnet, imds, 'new_fc', 'MiniBatchSize', 32);
                                        end