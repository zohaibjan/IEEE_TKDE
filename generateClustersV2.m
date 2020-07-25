function [genClusters,eachClass] = generateClustersV2(train , params)
totalClusters = 1;
genClusters = {};
dataClasses = unique(train(:,end))';
avgClass = [];
try
if totalClusters == 1
    [clusterIds, C, sum, D] = kmeans(train, 1, 'MaxIter', 24000);
    genClusters{totalClusters}.train = train(find(clusterIds == 1), :);
    genClusters{totalClusters}.centroid = C(1,:);
    totalClusters = totalClusters + 1;
end

for i=1:length(dataClasses)
    Xtrain = train(train(:,end) == dataClasses(i),:);
    if size(Xtrain,1) <= 2
        params.eachClass = 1;
    else
    rng('default');  % For reproducibility
    eva = evalclusters(Xtrain,'kmeans','silhouette','KList',[1:20]);
    params.eachClass = eva.OptimalK;
    avgClass(i) = params.eachClass;
    end
    [clusterIds, C, sum, D] = kmeans(zscore(Xtrain), params.eachClass, 'MaxIter', 24000);
    for j=1:params.eachClass
        genClusters{totalClusters}.train = Xtrain(find(clusterIds == j), :);
        genClusters{totalClusters}.centroid = C(j,:);
        totalClusters = totalClusters + 1;
    end
end
catch exc 
    disp(fprintf('\nProblem with %s \n%s\n', params.p_name, exc.identifier));
end
eachClass = mean(avgClass);
end


 