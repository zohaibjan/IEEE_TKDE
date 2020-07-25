function [accuracy, fMeasures, decisionMatrix, BCP]=accuracyOfPSO(classifiers, chromosome, testData)
c = find(chromosome);
X = testData(:, 1:end-1);
y = testData(:,end);
decisionMatrix = ones(length(testData(:,1)), length(c));
for i=1:length(c)
    try
        if strcmp(classifiers{1,i}.name, 'SVM') == 1
            decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'KNN') == 1
            decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DT') == 1
            decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'NB') == 1
            decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DISCR') == 1
            decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
            decisionMatrix(:,i) = getNNPredict(classifiers{1,i}.model, X);
        end
    catch ME
        disp(sprintf('IN ACCURACY OF PSO: %s',ME.identifier));
        continue
    end
end
decisionMatrix = mode(decisionMatrix, 2);
accuracy = mean(decisionMatrix == y);
fMeasures = confusionmatStats(y, decisionMatrix);
end