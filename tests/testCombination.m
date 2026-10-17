function tests = testCombination
tests = functiontests(localfunctions);
end
function testBKSAndMajority(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
trainLabels = [1;3;3;2];
trainPatterns = [1,2;1,2;3,3;2,2];
testPatterns = [1,2;2,2;3,1];
for k = 1:2
    training(k).Abstract_level_output = trainPatterns(:,k);
    training(k).Measurment_level_output = zeros(4,3);
    training(k).ConfusionMatrix = eye(3);
    testing(k).Abstract_level_output = testPatterns(:,k);
    testing(k).Rank_level_output = zeros(3,3);
    testing(k).Measurment_level_output = zeros(3,3);
end
% Seen pattern [1,2] has labels 1 and 3: mode deterministically returns 1.
verifyEqual(testCase,CombineCLFs(10,training,testing,2,3,4,3,trainLabels),[1,2,1]);
verifyEqual(testCase,CombineCLFs(1,training,testing,2,3,4,3,trainLabels),[1,2,1]);
end
