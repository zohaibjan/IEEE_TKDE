function genClusters = clusterData(Xtrain, Ytrain , params)
    data = [Xtrain, Ytrain];
    totalClusters = 1;
    genClusters = {};
    dataClasses = unique(Ytrain)';
    allClasses = ones(1,length(dataClasses))*params.eachClass; 
    k = 1;
    while mean(allClasses) ~= 0        
        if size(Xtrain,1) > k
            [clusterIds, C, sum, D] = kmeans(zscore(Xtrain), k, 'MaxIter', 2400);  
            if k == 1
                genClusters{totalClusters}.train = data(clusterIds == k,:); % indexes of clusters
                genClusters{totalClusters}.centroid = C(k,:); % indexes of clusters
                genClusters{totalClusters}.balance = getBalance(Ytrain(clusterIds == k));
                y_prime = Ytrain(clusterIds == k);
                classes = unique(y_prime);
                noOfClasses = length(unique(y_prime));
                genClusters{totalClusters}.noOfClasses = noOfClasses;
                totalClusters = totalClusters + 1;
            else
                for j=1:k
                    samples = Ytrain(find(clusterIds == j), :);
                    strongClass = mode(samples);
                    if allClasses(dataClasses == strongClass) ~= 0
                        allClasses(dataClasses == strongClass) = allClasses(dataClasses == strongClass) - 1;
                        genClusters{totalClusters}.train = data(clusterIds == j,:); % indexes of clusters
                        genClusters{totalClusters}.centroid = C(j,:); % indexes of clusters
                        genClusters{totalClusters}.balance = getBalance(Ytrain(clusterIds == j));
                        y_prime = Ytrain(clusterIds == j);
                        classes = unique(y_prime);
                        noOfClasses = length(unique(y_prime));
                        genClusters{totalClusters}.noOfClasses = noOfClasses;
                        totalClusters = totalClusters + 1;
                    end
                end
            end
            k = k + 1;
        else
            break;
        end
    end
end


% 'MaxIter',500, 'Replicates',5,  'dist','sqeuclidean'