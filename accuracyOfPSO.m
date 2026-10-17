function [accuracy, fMeasures, decisionMatrix, BCP] = accuracyOfPSO(classifiers, chromosome, testData)
%ACCURACYOFPSO Score the actual selected model indices without fabricated votes.
if numel(chromosome) ~= numel(classifiers) || any(~ismember(chromosome(:), [0 1]))
    error('Research:InvalidSelection', 'Supply one binary flag per classifier.');
end
selected = classifiers(logical(chromosome));
decisionMatrix = votePredictions(predictionMatrix(selected, testData(:, 1:end-1)));
y = numericLabels(testData(:, end));
accuracy = mean(decisionMatrix == y);
BCP = struct('SVM', 0, 'ANN', 0, 'DT', 0, 'KNN', 0, 'NB', 0, 'DISCR', 0);
for i = 1:numel(selected)
    name = selected{i}.name;
    if ~isfield(BCP, name)
        BCP.(name) = 0;
    end
    BCP.(name) = BCP.(name) + 1;
end
% This historical optional report still requires confusionmatStats on the path.
fMeasures = confusionmatStats(y, decisionMatrix);
end
