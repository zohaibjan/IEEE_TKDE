%%collects classifiers and stores them in an array.
function [classifier] = trainSVM(X, y, valX, valY)
radial=templateSVM('KernelFunction','rbf','IterationLimit',50000,'Standardize',true);
linear = templateSVM('KernelFunction','linear','IterationLimit',50000,'Standardize',true);
try 
    rbf.name = 'SVM';
    rbf.model = fitcecoc(X, y, 'learners', radial, 'ClassNames',[unique(y)]);
    
    lin.name = 'SVM';
    lin.model = fitcecoc(X, y, 'learners', linear, 'ClassNames',[unique(y)]);
    
    
    predictRbf = predict(rbf.model, valX);
    predictLin = predict(lin.model, valX);
    
    accRbf = mean(predictRbf == valY);
    accLin = mean(predictLin == valY); 
    
    if accRbf > accLin
        classifier = rbf;
    elseif accLin > accRbf
        classifier = lin;
    else
        classifier = rbf;
    end
catch exc
    disp(sprintf('something happened in training %s \n', exc.identifier));
end

end