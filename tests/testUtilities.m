function tests = testUtilities
% Small synthetic regression checks; no dataset or learner toolbox required.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testCase.TestData.originalPath = path;
testCase.TestData.root = fileparts(fileparts(mfilename('fullpath')));
addpath(testCase.TestData.root, '-begin');
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function testMinMaxWithConstantFeature(testCase)
actual = dataNormalize([1 7; 3 7; 5 7], 1);
verifyEqual(testCase, actual, [0 0; 0.5 0; 1 0], 'AbsTol', 1e-12);
end

function testZeroMeanAndConstantZScore(testCase)
actual = dataNormalize([-1 7; 0 7; 1 7], 2);
verifyEqual(testCase, actual, [-1 0; 0 0; 1 0], 'AbsTol', 1e-12);
end

function testOrdinaryZScore(testCase)
actual = dataNormalize([2; 4; 6], 2);
verifyEqual(testCase, actual, [-1; 0; 1], 'AbsTol', 1e-12);
end

function testEmptyInput(testCase)
verifyEqual(testCase, dataNormalize([], 1), []);
verifyEqual(testCase, dataNormalize([], 2), []);
end

function testCsvHeaderAndAppend(testCase)
originalDir = pwd;
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() restoreDirectory(originalDir, folder));
cd(folder);
fields = {'p_name', 'eachClass', 'runs', 'nonOptimized_Accuracy', ...
    'nonOptimized_stdDEV', 'optimized_Accuracy', 'optimized_stdDEV', ...
    'accuracy', 'stdDev', 'majVote', 'mstd_dev', 'clusterBased', ...
    'cstd_dev', 'e_mVote', 'e_mVote_dev', 'allMVote', 'allMVote_dev', ...
    'avgAccuracy', 'stdDevWithBCS', 'avgAccuracy_without_BCS', ...
    'stdDevWithoutBCS', 'originalSize', 'sizeAfterBCS', 'addedDiv'};
for i = 1:numel(fields)
    result.(fields{i}) = 0.5;
end
result.p_name = 'synthetic';
saveResults(result);
saveResults(result);
rows = regexp(strtrim(fileread('results.csv')), '\r?\n', 'split');
verifyEqual(testCase, numel(rows), 3);
verifyEqual(testCase, numel(strfind(fileread('results.csv'), 'Data Set')), 1);
verifyTrue(testCase, ~isempty(strfind(rows{2}, 'synthetic')));
verifyEqual(testCase, rows{2}, rows{3});
end

function restoreDirectory(originalDir, folder)
cd(originalDir);
rmdir(folder, 's');
end
