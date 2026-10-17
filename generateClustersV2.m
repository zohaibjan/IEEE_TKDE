function [genClusters, eachClass] = generateClustersV2(train, params)
%GENERATECLUSTERSV2 Feature-only candidates with bounded silhouette search.
% params is retained for compatibility; this path chooses K from training features.
if isempty(train) || size(train, 2) < 2 || any(~isfinite(train(:)))
    error('Research:InvalidTrainingData', 'Supply finite labelled training rows.');
end
classes = unique(train(:, end));
genClusters = {struct('train', train, 'centroid', mean(train(:, 1:end-1), 1))};
classCounts = ones(numel(classes), 1);
for i = 1:numel(classes)
    rows = train(train(:, end) == classes(i), :);
    features = rows(:, 1:end-1);
    maxK = min([20, size(features, 1) - 1, size(unique(features, 'rows'), 1)]);
    k = 1;
    if maxK >= 2
        evaluation = evalclusters(features, 'kmeans', 'silhouette', 'KList', 2:maxK);
        if isfinite(evaluation.OptimalK)
            k = evaluation.OptimalK;
        end
    end
    classCounts(i) = k;
    if k == 1
        ids = ones(size(features, 1), 1);
    else
        ids = kmeans(dataNormalize(features, 2), k, 'MaxIter', 24000);
    end
    for j = 1:k
        members = rows(ids == j, :);
        if isempty(members)
            continue;
        end
        genClusters{end + 1} = struct('train', members, ...
            'centroid', mean(members(:, 1:end-1), 1));
    end
end
eachClass = mean(classCounts);
end
