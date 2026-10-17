function obj=classifierSelectionPSO(classifierList, testData, solverOptions)

if nargin < 3
    solverOptions = [];
end

try

    allPredictions = psoPredict(classifierList, testData);
    %set optimization function to PSOAF
    fun = @PSOAF;
    options2 = optimoptions('particleswarm','SwarmSize',100,...
        'MaxIterations', 100, 'MaxStallIterations', 5);
    if ~isempty(solverOptions)
        options2 = solverOptions;
    end
    lb=zeros(1,length(classifierList));
    ub=ones(1,length(classifierList));
    [best,fval,exitflag,output]=particleswarm(fun, length(classifierList),lb,ub,options2);
    obj.chromosome=double(selectionMask(best, 0.4));
    obj.fval=fval;
    obj.output=output;
catch exc
    rethrow(exc);
end


%% OBJECTIVE FUNCTION
    function error=PSOAF(c)
        % BINARIZE THE CLASSIFIER SELECTION
        c = selectionMask(c, 0.4);
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