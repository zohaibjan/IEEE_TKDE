function labels = fusionPSO(classifiers, testData)
%FUSIONPSO Return a majority prediction for each sample.
labels = votePredictions(predictionMatrix(classifiers, testData(:, 1:end-1)));
end
