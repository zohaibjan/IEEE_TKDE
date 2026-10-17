function labels = predictLabels(classifier, X)
%PREDICTLABELS One validated prediction per row; prediction failures propagate.
if any(strcmp(classifier.name, {'ANN', 'RBFNN'}))
    values = getNNPredict(classifier.model, X);
elseif strcmp(classifier.name, 'CNN')
    values = getCNNPred(classifier.model, X);
else
    values = predict(classifier.model, X);
end
labels = numericLabels(values);
if numel(labels) ~= size(X, 1)
    error('Research:PredictionSize', 'The model must return one label per sample.');
end
end
