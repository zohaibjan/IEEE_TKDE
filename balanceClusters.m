function [balancedClusters, centroids] = balanceClusters(clusters, train)
noOfClasses = unique(train(:,end));
b_index = 1;
balancedClusters = {};
centroids = {};
for i=1:size(clusters,2)
    %% balance clusters of that class
    balancedClusters{1,b_index} = clusters{1, i}.train;
    majorClass = unique(clusters{1, i}.train(:,end));
    centroids{1,i} = clusters{1, i}.centroid;
    
    if i==1
        b_index = b_index + 1;
        continue; 
    end
    for j=1:length(noOfClasses)
        if majorClass ~= noOfClasses(j)
            records = length(clusters{1, i}.train);
            toAdd = train(train(:,end) == noOfClasses(j),:);
            closestToCentroid = [];
            for k=1:size(toAdd,1)
                closestToCentroid(k,1) = norm(clusters{1, i}.centroid - (toAdd(k,:)));
            end
            [sorted, sorted_ids] = sort(closestToCentroid);
            toAdd = toAdd(sorted_ids, :);
            
            if size(toAdd,1) >= records
                balancedClusters{1,b_index} = [balancedClusters{1,b_index}; toAdd(1:records,:)];
            else
                balancedClusters{1,b_index} = [balancedClusters{1,b_index}; toAdd];
            end
        end
        
    end
    b_index = b_index + 1;
end
end