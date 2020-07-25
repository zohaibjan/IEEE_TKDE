function predictions = psoPredict(classifiers, testData)
    X = testData(:,1:end-1);
    predictions = ones(length(testData(:,end)), length(classifiers));
    for i=1:length(classifiers)            
        try
            if strcmp(classifiers{1,i}.name, 'ANN') == 0
                 predictions(:,i) = predict(classifiers{1,i}.model, X);
            elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
                 predictions(:,i) = getNNPredict(classifiers{1,i}.model, X);
            end
        catch ME
            disp(sprintf('IN psoPredict: %s',ME.identifier));
            continue
        end
    end
    
end