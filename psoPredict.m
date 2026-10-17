function predictions = psoPredict(classifiers, testData)
%PSOPREDICT Prediction errors propagate rather than becoming synthetic votes.
predictions = predictionMatrix(classifiers, testData(:, 1:end-1));
end
