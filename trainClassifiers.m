%%collects classifiers and stores them in an array.
function [classifiers] = trainClassifiers(X, y, valX, valy, params)
maxTreeSize = length(unique(y)); 
def.seed=100;
%% randomize parameters
def.range=[10 30;               % hidden neuron NN
    1 9;                        % training function NN
    100 5000;                   % NN epochs
    1 4;                        % SVM training functions
    1 2;                        % NB training function
    1 10;                       % kNN num neighbors
    maxTreeSize maxTreeSize*10; % DT MinleafSize  TRY 20 or other
    1 2                         % DA train function
    5000 10000];                 % SVM iteration limit
p=floor(def.range(:,1)+(def.range(:,2)-def.range(:,1))*rand);

%% Templates for CLassifiers
t = templateTree('minleafsize', p(7));
knn = templateKNN('NumNeighbors', p(6));
svm=templateSVM('KernelFunction','rbf','IterationLimit',p(9),'Standardize',true);

classifiers = {};
index = 1;
try
    for i = 1:length(params.classifiers)
        learner = params.classifiers{i};
        %% SWITCH
        switch learner
            
            %% KNN CLASSIFIER
            case 'KNN'
                classifiers{index}.name = 'KNN';
                classifiers{index}.model = fitcknn(X, y, 'NumNeighbors', p(6));
                index = index + 1;
                
                %% SVM CLASSIFIER
            case 'SVM'
                classifiers{index} = trainSVM(X, y, valX, valy);
                index = index + 1;
                
                %% Naive Bayes CLASSIFIER
            case 'NB'
                c=unique(y); v=zeros(1,length(c));
                for i=1:length(c)
                    v(i)=sum(y==c(i));
                end
                if(min(v)<2)
                    continue;
                end
                classifiers{index}.name = 'NB';
                classifiers{index}.model = fitcnb(X, y, 'distribution', 'kernel');
                index = index + 1;
                
                %% Discriminant Analysis CLASSIFIER
            case 'DISCR'
                c=unique(y); v=zeros(1,length(c));
                for i=1:length(c)
                    v(i)=sum(y==c(i));
                end
                if(min(v)<2)
                    continue;
                end
                if size(X,1) < 5
                    continue
                end
                classifiers{index}.name = 'DISCR';
                classifiers{index}.model = fitcdiscr(X, y, 'discrimtype','diaglinear');
                index = index + 1;
                
                %% Decision Tree
            case 'DT'
                classifiers{index}.name = 'DT';
                classifiers{index}.model = fitctree(X, y);
                index = index + 1;
                
                %% Neural Network
            case 'ANN'
                classifiers{index}.name = 'ANN';
                classifiers{index}.model = trainNN(X, y);
                index = index + 1;
                
            case 'RBFNN'
                classifiers{index}.name = 'ANN';
                classifiers{index}.model = trainRBFNN(X, y, params, p);
                index = index + 1;
                %% LP BOOST
            case 'LPBOOST'
                classifiers{index}.name = 'LPBOOST';
                classifiers{index}.model =  fitcensemble(X,y,'Method', 'Lpboost', 'learners', t); %% NOT GOOD
                index = index + 1;
                
                %% BAG
            case 'BAG'
                classifiers{index}.name = 'BAG';
                classifiers{index}.model =  fitcensemble(X,y,'Method','bag', 'learners', t);
                index = index + 1;
                
                %% Subspace
            case 'SUBSPACE'
                classifiers{index}.name = 'SUBSPACE';
                classifiers{index}.model =  fitcensemble(X,y,'Method','subspace', 'learners', knn);
                index = index + 1;
                
                %% Total BOOST
            case 'TOTALBOOST'
                classifiers{index}.name = 'TOTALBOOST';
                classifiers{index}.model =  fitcensemble(X,y,'Method','totalboost', 'learners', t); %% NOT GOOD
                index = index + 1;
                
                %% Total BOOST
            case 'ADABOOST'
                if length(unique(y)) > 2
                    classifiers{index}.name = 'ADABOOST';
                    classifiers{index}.model =  fitcensemble(X,y, 'Method', 'AdaBoostM2', 'learners', t);
                    index = index + 1;
                elseif length(unique(y)) == 2
                    classifiers{index}.name = 'ADABOOST';
                    classifiers{index}.model =  fitcensemble(X,y, 'Method', 'AdaBoostM1', 'learners', t);
                    index = index + 1;
                end
                
            otherwise
                warning('Unknown Classifier')
        end
    end
catch exc
    disp(sprintf('something happened in trainingClassifiers %s \n', exc.identifier));
end

end