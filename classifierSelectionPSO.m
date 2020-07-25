function obj=classifierSelectionPSO(classifierList, testData)
warning('off','all');
try
    warning('off','all')
    allPredictions = psoPredict(classifierList, testData);
    %set optimization function to PSOAF
    fun = @PSOAF;
    options2 = optimoptions('particleswarm','SwarmSize',100,...
        'MaxIterations', 100, 'MaxStallIterations', 5);
    lb=zeros(1,length(classifierList));
    ub=ones(1,length(classifierList));
    [best,fval,exitflag,output]=particleswarm(fun, length(classifierList),lb,ub,options2);
    obj.chromosome=round(best);
    obj.fval=fval;
    obj.output=output;
catch exc
    disp(sprintf('problem with %s', exc.identifier));
end


%% OBJECTIVE FUNCTION
    function error=PSOAF(c)
        % BINARIZE THE CLASSIFIER SELECTION
        c = c > 0.4;
        c = find(c);
        
        %% CALCULATE THE ACCURACY
        decisionMatrix = ones(length(testData(:,end)), length(c));
        for i=1:length(c)
            decisionMatrix(:,i) = allPredictions(:, c(i)) ;
        end
        decisionMatrix = mode(decisionMatrix, 2);
        error = mean(decisionMatrix ~= testData(:,end));
    end
end