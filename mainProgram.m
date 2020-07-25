function program = mainProgram()

addpath(genpath('P-Data'));
Problem={'adult','australian','balance','banknote',...
    'breast-cancer-wisconsin','ecoli','haberman','ionosphere','iris'....
    'liver','page-blocks','pima_diabetec','segment','sonar','statimag',...
    'teaching','thyroid','vehicle','vowel','wdbc','wine','DNA',...
    'fertility','heart','letter-recognition','hepatitis','bupa',...
    'transfusion','zoo','hayes-roth'};




% Problem = {'wine'};

%% Model SETTINGS
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
params.classifiers = {'KNN' 'DISCR', 'SVM','ANN','DT'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};

%% MAIN LOOP
parfor i = 1:length(Problem)
    for eachClass = 2:10
        p_name = Problem{i};
        results = runTraining(p_name, params, eachClass);
        results.p_name = p_name;        
        results.eachClass = eachClass;
        saveResults(results);
    end
end
end

