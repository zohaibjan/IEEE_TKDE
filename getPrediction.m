function preds=getPrediction(classifier, X)
try
    if strcmp(classifier.name, 'SVM') == 1
        preds = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'KNN') == 1
        preds = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'DT') == 1
        preds = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'NB') == 1
        preds = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'DISCR') == 1
        preds= predict(classifier.model, X);
    elseif strcmp(classifier.name, 'ANN') == 1
        preds = getNNPredict(classifier.model, X);
    elseif strcmp(classifier.name, 'CNN') == 1
        preds = getCNNPred(classifier.model, X);
    elseif strcmp(classifier.name, 'RBFNN') == 1
        preds = getNNPredict(classifier.model, X);
    end
catch ME
    disp(sprintf('Problem in getPrediction: %s',ME.identifier));
end


end