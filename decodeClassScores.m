function labels = decodeClassScores(scores, classLabels)
%DECODECLASSSCORES Map output rows back to the training label vocabulary.
classes = numericLabels(classLabels);
if ~isnumeric(scores) || isempty(scores) || ~ismatrix(scores) || ...
        ~isreal(scores) || any(~isfinite(scores(:))) || size(scores, 1) ~= numel(classes) || ...
        numel(unique(classes)) ~= numel(classes)
    error('Research:InvalidScores', 'Scores and unique training labels must have matching rows.');
end
[~, indices] = max(scores, [], 1);
labels = classes(indices);
labels = labels(:);
end
