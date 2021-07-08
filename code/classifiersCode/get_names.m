function feature_predictor_names1 = get_names(t)
    [~,dimy]=size(t.features);
fprintf('%s%d\n', "Inizio dimensioni",dimy);
for index=1:dimy
    feature_predictor_names1=[feature_predictor_names1,strcat("features_",string(index))];
end
fprintf('%s%d\n', "nomi predittori",length(feature_predictor_names1));
end