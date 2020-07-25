function fusion = fusionPSO(classifiers, testData)
tempPredict = {};
index = 1;
X = testData(:,1:end-1);
y = testData(:,end);
for i=1:length(classifiers)
    try
        if strcmp(classifiers{1,i}.name, 'SVM') == 1
            tempPredict{index} = predict(classifiers{1,i}.model, X);
            index = index + 1;
            
        elseif strcmp(classifiers{1,i}.name, 'DT') == 1
            tempPredict{index} = predict(classifiers{1,i}.model, X);
            index = index + 1;
            
        elseif strcmp(classifiers{1,i}.name, 'DISCR') == 1
            tempPredict{index} = predict(classifiers{1,i}.model, X);
            index = index + 1;
            
        elseif strcmp(classifiers{1,i}.name, 'KNN') == 1
            tempPredict{index} = predict(classifiers{1,i}.model, X);
            index = index + 1;
            
        elseif strcmp(classifiers{1,i}.name, 'NB') == 1
            tempPredict{index} = predict(classifiers{1,i}.model, X);
            index = index + 1;
            
        elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
            tempPredict{index} = getNNPredict(classifiers{1,i}.model, X);
            index = index + 1;
        end
    catch ME
        disp('in PSO FUSION');
        continue
    end
end
%% WHEN ALL IS DONE SAFELY
decisionMatrix = ones(length(testData(:,1)), length(tempPredict));
for j = 1:length(tempPredict)
    decisionMatrix(:,j) = cell2mat(tempPredict(j));
end
fusion = mode(decisionMatrix,2);
end

