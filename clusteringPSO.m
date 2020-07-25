function obj=clusteringPSO(allClusters, testData, params)
warning('off','all');
allPredictions = zeros(length(testData(:,end)), length(allClusters));
for j = 1:length(allClusters)
    all = [];
    all = trainClassifiers(allClusters{1,j}(:,1:end-1), allClusters{1,j}(:,end), params);
    allPredictions(:, j) = fusionPSO(all, testData);
end
%set optimization function to PSOAF
fun = @PSOAF;
options2 = optimoptions('particleswarm','SwarmSize',50);
lb=zeros(1,size(allPredictions,2));
ub=ones(1,size(allPredictions,2));
[best,fval,exitflag,output]=particleswarm(fun, size(allPredictions,2),lb,ub,options2);
obj.chromosome=round(best);
obj.fval=fval;
obj.output=output;

%% OBJECTIVE FUNCTION
function error=PSOAF(c)
% BINARIZE THE CLASSIFIER SELECTION
c = c > 0.6;
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
