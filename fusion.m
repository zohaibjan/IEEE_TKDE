function acc = fusion(classifiers, data)
%FUSION Score validated model votes against the original numeric labels.
predictions = predictionMatrix(classifiers, data(:, 1:end-1));
acc = mean(votePredictions(predictions) == numericLabels(data(:, end)));
end
