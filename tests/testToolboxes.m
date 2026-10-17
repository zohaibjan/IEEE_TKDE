function tests = testToolboxes
% Small synthetic runtime checks; no paper dataset or accuracy benchmark.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testCase.TestData.originalPath = path;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'tests', 'fixtures'), '-begin');
rng(42, 'twister');
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function [X, y] = syntheticData()
X = [randn(16, 6) - 1; randn(16, 6) + 1];
y = [zeros(16, 1); 10 * ones(16, 1)];
end

function testActualSvmAndTreePredictions(testCase)
[X, y] = syntheticData();
a.name = 'SVM';
a.model = fitcecoc(X, y, 'Learners', templateSVM('Standardize', true));
b.name = 'DT'; b.model = fitctree(X, y);
preds = predictionMatrix({a, b}, X);
verifySize(testCase, preds, [32 2]);
verifyTrue(testCase, all(ismember(preds(:), [0 10])));
end

function testActualNeuralNetworkLabelVocabulary(testCase)
[X, y] = syntheticData();
net = trainNN(X, y);
labels = getNNPredict(net, X(1:3, :));
verifySize(testCase, labels, [3 1]);
verifyTrue(testCase, all(ismember(labels, [0 10])));
verifyEqual(testCase, net.userdata.classLabels, [0; 10]);
end

function testActualParticleSwarmObjectiveMatchesReturnedMask(testCase)
a.name = 'DT'; a.model = FixedClassifier([0; 0; 10; 10]);
b.name = 'DT'; b.model = FixedClassifier([10; 10; 0; 0]);
models = {a, b};
data = [1 0; 2 0; 3 10; 4 10];
options = optimoptions('particleswarm', 'SwarmSize', 12, 'MaxIterations', 6, ...
    'MaxStallIterations', 3, 'Display', 'off', 'UseParallel', false);
result = classifierSelectionPSO(models, data, options);
verifyEqual(testCase, numel(result.chromosome), 2);
verifyTrue(testCase, any(result.chromosome));
actual = mean(votePredictions(psoPredict(models(logical(result.chromosome)), data)) ~= data(:, end));
verifyEqual(testCase, result.fval, actual, 'AbsTol', 1e-12);
end

function testActualSilhouetteClusterGeneration(testCase)
[X, y] = syntheticData();
[clusters, k] = generateClustersV2([X y], struct());
verifyGreaterThanOrEqual(testCase, k, 1);
verifyLessThanOrEqual(testCase, k, 15);
for i = 1:numel(clusters)
    verifySize(testCase, clusters{i}.centroid, [1 6]);
    verifyTrue(testCase, all(isfinite(clusters{i}.centroid)));
end
end
