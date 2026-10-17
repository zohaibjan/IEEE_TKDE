function [balancedClusters, centroids] = balanceClusters(clusters, train)
%BALANCECLUSTERS Add absent classes using feature-only Euclidean distances.
classes = unique(train(:, end));
balancedClusters = cell(1, numel(clusters));
centroids = cell(1, numel(clusters));
for i = 1:numel(clusters)
    base = clusters{i}.train;
    if isempty(base) || size(base, 2) ~= size(train, 2)
        error('Research:InvalidCluster', 'Each cluster must contain labelled training rows.');
    end
    centroid = mean(base(:, 1:end-1), 1);
    centroids{i} = centroid;
    balancedClusters{i} = base;
    if i == 1
        continue; % The first cluster is the historical full-training baseline.
    end
    present = unique(base(:, end));
    for j = 1:numel(classes)
        if any(present == classes(j))
            continue;
        end
        candidates = train(train(:, end) == classes(j), :);
        offsets = bsxfun(@minus, candidates(:, 1:end-1), centroid);
        [~, order] = sort(sum(offsets .^ 2, 2));
        count = min(size(base, 1), size(candidates, 1));
        balancedClusters{i} = [balancedClusters{i}; candidates(order(1:count), :)];
    end
end
end
