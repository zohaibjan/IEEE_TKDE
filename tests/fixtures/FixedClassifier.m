classdef FixedClassifier
    %FIXEDCLASSIFIER Synthetic predict-compatible model for regression tests.
    properties
        Labels
        ShouldFail = false
    end
    methods
        function model = FixedClassifier(labels, shouldFail)
            model.Labels = labels;
            if nargin > 1
                model.ShouldFail = shouldFail;
            end
        end
        function labels = predict(model, X)
            if model.ShouldFail
                error('Research:MockFailure', 'Synthetic prediction failed.');
            end
            labels = model.Labels(X(:, 1));
        end
    end
end
