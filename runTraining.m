function results = runTraining(p_name , params, class)
warning('off','all');
params.eachClass = class;
nonOptimized_Accuracy = [];
optimized_Accuracy = [];
params.p_name = p_name;
data = load(p_name);
data = [data.X, data.y];
cvFolds = cvpartition(data(:,end), 'KFold', params.numOfFolds);

%% ITERATE OVER THE NUMBER OF FOLDS
for f=1:params.numOfFolds
    classifierIndex = 1;
    classifiers = {};
    
    idx = cvFolds.test(f);
    trainData = data(~idx,:);
    testData = data(idx,:);
    cv = cvpartition(trainData(:,end), 'holdout', 0.1);
    idxs = cv.test;
    validationData = trainData(idxs,:);
    trainData = trainData(~idxs, :);
    
    trainX = trainData(:, 1:end-1);
    trainy = trainData(:, end);
    
    testX = testData(:, 1:end-1);
    testy = testData(:, end);
    
    valX = validationData(:, 1:end-1);
    valy = validationData(:, end);
    
    trainX(isnan(trainX)) = -1;
    testX(isnan(testX)) = -1;
    valX(isnan(valX)) = -1;
    
    allClusters = generateClustersV2([trainX, trainy], params);
    [balancedClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
    
    for c=balancedClusters
        X = c{1,1}(:, 1:end-1);
        y = c{1,1}(:, end);
        all = trainClassifiers(X, y, valX, valy, params);
        if size(all,1) < 1
            continue
        end
        for temp = 1:length(all)
            classifiers{classifierIndex} = all{1,temp};
            classifierIndex = classifierIndex + 1;
        end
    end
    
    psoEnsemble = classifierSelectionPSO(classifiers, [valX, valy]);
    psoEnsemble = find(psoEnsemble.chromosome);
    selectedClassifiers = {};
    
    for i=1:length(psoEnsemble)
        selectedClassifiers{1,i} = classifiers{1, psoEnsemble(i)};
    end
    
    nonOptimized_Accuracy(f) = fusion(classifiers, [testX, testy]);
    optimized_Accuracy(f) = fusion(selectedClassifiers, [testX, testy]);
end
results.nonOptimized_Accuracy = mean(nonOptimized_Accuracy);
results.optimized_Accuracy = mean(optimized_Accuracy);
results.nonOptimized_stdDEV = std(nonOptimized_Accuracy);
results.optimized_stdDEV = std(optimized_Accuracy);
end

