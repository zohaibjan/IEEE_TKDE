function labels = getPrediction(classifier, X)
%GETPREDICTION Compatibility entry point for validated label prediction.
labels = predictLabels(classifier, X);
end
