function saveResults(results)
  fid = [];
    if (exist([pwd filesep 'results.csv'], 'file') == 0)
        fid = fopen([pwd filesep 'results.csv'], 'w');
        if fid == -1
            error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
        end
        fileCleanup = onCleanup(@() fclose(fid));
        fprintf(fid, '%s, %s, %s, %s, %s, %s\n', ...
            'Data Set','Avg Accuracy', 'Std. Dev', 'Optimized Acc', 'Std. Dev', 'K'...
            );
    elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
        fid = fopen([pwd filesep 'results.csv'], 'a');
        if fid == -1
            error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
        end
        fileCleanup = onCleanup(@() fclose(fid));
    end
    fprintf(fid, '%s, ', results.p_name);
    fprintf(fid, '%f, %f, %f, %f, %f\n', ...
         results.nonOptimized_Accuracy, results.nonOptimized_stdDEV,...
         results.optimized_Accuracy, results.optimized_stdDEV, results.eachClass ...
        );

end



