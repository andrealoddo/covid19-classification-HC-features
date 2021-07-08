    if saveTables == 1
        headings =  ['\\documentclass[12pt,italian]{article}\n',...savetable
            '\\usepackage{graphicx}\n',...
            '\\usepackage{longtable}\n',...
            '\\parskip 0.1in\n',...
            '\\oddsidemargin -1in\n',...
            '\\evensidemargin -1in\n',...
            '\\topmargin -0.5in\n',...
            '\\textwidth 6.8in\n',...
            '\\textheight 9.9in\n',...
            '\\usepackage{fancyhdr}\n',...
            '\\usepackage{booktabs}\n',...
            '\\usepackage{multirow}\n',...
            '\\usepackage{amsmath}\n',...
            '\\begin{document}\n',...
            '\\begin{tiny}\n'];

        closing = '\\end{tiny} \n \\end{document}';
        %Collect and write results
        for dt = 1:numel(datasets)
            %Write results in a LaTeX table    
            
            for sp = 1:numel(splits{dt})
                %Load labels
                if strcmp(datasetsname{dt}, 'CNMC')
                    string_split = '';
                else
                    string_split = splits{dt}{sp};
                end
                destinationPerf = [perfPath sep,...
                    'Performances___',...
                    datasetsname{dt} '___',...
                    string_split '.tex'];
                pFile = fopen(destinationPerf, 'w');
                fprintf(pFile, headings);
                for fs = 1:numel(featselector)
                    for sel = 1:numel(selection)
                        if selection(sel)<100 && size(DBTrain,2) > 10
                            selected = featureSelection(featselector{fs}, DBTrain, labels, selection(sel));
                            string_selection = [featselector{fs} '___' num2str(selection(sel)) '___'];
                        else
                            selected = [];
                            string_selection = '';
                        end
                        for gl = 1:numel(graylevel)
                            for pp = 1:numel(prepro)
                                for pop = 1:numel(postpro)         
                                    %Write table
                                    fprintf(pFile, '\\begin{longtable}{l');
                                    %fprintf(pFile, '\\begin{tabular}{l');
                                    for cla = 1:(numel(classifier)*5 + 1)
                                        fprintf(pFile, 'c');
                                    end
                                    fprintf(pFile, '}\n');
                                    fprintf(pFile, '\\toprule\n');
                                    %Heading
                                    fprintf(pFile, '\\multicolumn{%d}{c}{Dataset=%s selection=%s\\%% prepro= %s postpro= %s, gl= %s} \\\\ \n', numel(classifier)*5+1, datasetsname{dt}, string_selection, prepro{pp}, postpro{pop}, num2str(graylevel(gl)));
                                    fprintf(pFile, '\\toprule\n');
                                    fprintf(pFile, 'Descriptor & \\multicolumn{%d}{c}{Classifier} \\\\ \n', numel(classifier)*5);
                                    for cla = 1:numel(classifier)
                                        fprintf(pFile, '& \\multicolumn{5}{c}{%s} ', classifier{cla});
                                    end
                                    fprintf(pFile, '\\\\ \n');
                                    for ni = 1:numel(classifier)
                                        fprintf(pFile, '& A & P & R & S & F1 ');
                                    end
                                    fprintf(pFile, '\\\\ \n');
                                    fprintf(pFile, '\\midrule\n');

                                    %Data
                                    sumA = zeros(numel(classifier),1);
                                    sumP = zeros(numel(classifier),1);
                                    sumR = zeros(numel(classifier),1);
                                    sumS = zeros(numel(classifier),1);
                                    sumF1 = zeros(numel(classifier),1);
                                    for dsc_set = 1:numel(descriptors_sets)
                                        if contains( descriptors_sets{dsc_set}, '_' )
                                            desc_write = erase( descriptors_sets_names{dsc_set}, '_' );
                                        else
                                            desc_write = descriptors_sets_names{dsc_set};
                                        end
                                        fprintf(pFile, '%s ', desc_write);
                                        for cla = 1:numel(classifier)                     
                                            %load retrieval results
                                            destinationResult = [classifPath '/',...
                                                datasetsname{dt} '___',...
                                                string_split '___',...
                                                descriptors_sets{dsc_set} '___',...
                                                string_selection,...
                                                num2str(graylevel(gl)) '___',...
                                                prepro{pp} '___',...
                                                postpro{pop} '___',...
                                                classifier{cla} '.mat'];
                                            if exist(destinationResult) ~= 0
                                                load(destinationResult, 'results');                       
                                                fprintf(pFile, '& %4.1f ', 100*zeroNaN(results.ACC));
                                                fprintf(pFile, '& %4.1f ', 100*zeroNaN(results.P));
                                                fprintf(pFile, '& %4.1f ', 100*zeroNaN(results.R));    
                                                fprintf(pFile, '& %4.1f ', 100*zeroNaN(results.TNR));
                                                fprintf(pFile, '& %4.1f ', 100*zeroNaN(results.F1)); 
                                                sumA(cla) = sumA(cla)+zeroNaN(results.ACC);
                                                sumP(cla) = sumP(cla)+zeroNaN(results.P);
                                                sumR(cla) = sumR(cla)+zeroNaN(results.R);
                                                sumS(cla) = sumS(cla)+zeroNaN(results.TNR);
                                                sumF1(cla) = sumF1(cla)+zeroNaN(results.F1);
                                            end
                                        end
                                        fprintf(pFile, '\\\\ \n');
                                    end
                                    fprintf(pFile, '\\hline\n');
                                    fprintf(pFile, 'AVG ');
                                    for cla = 1:numel(classifier)
                                        fprintf(pFile, '& %4.1f ', 100*(sumA(cla)/numel(descriptors_sets)));
                                        fprintf(pFile, '& %4.1f ', 100*(sumP(cla)/numel(descriptors_sets)));
                                        fprintf(pFile, '& %4.1f ', 100*(sumR(cla)/numel(descriptors_sets)));  
                                        fprintf(pFile, '& %4.1f ', 100*(sumS(cla)/numel(descriptors_sets)));
                                        fprintf(pFile, '& %4.1f ', 100*(sumF1(cla)/numel(descriptors_sets)));  
                                    end
                                    fprintf(pFile, '\\\\ \n');
                                    fprintf(pFile, '\\hline\n');
                                    fprintf(pFile, '\\bottomrule\n');
                                    fprintf(pFile, '\\end{longtable} \n');
                                    fprintf(pFile, '\n \\pagebreak \n');
                                end
                            end
                        end
                    end
                end
                if pFile ~= -1
                    fprintf(pFile, closing);
                    fclose(pFile);
                end
            end
        end
    end