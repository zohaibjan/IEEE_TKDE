function predictions = predictionMatrix(classifiers, X)
%PREDICTIONMATRIX Cache validated predictions; never substitute class-one votes.
if isempty(classifiers) || isempty(X)
    error('Research:EmptyEnsemble', 'A nonempty classifier pool and input are required.');
end
predictions = zeros(size(X, 1), numel(classifiers));
for i = 1:numel(classifiers)
    predictions(:, i) = predictLabels(classifiers{i}, X);
end
end
