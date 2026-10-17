function labels = votePredictions(predictions)
%VOTEPREDICTIONS Majority vote; exact ties choose the smallest numeric label.
if ~isnumeric(predictions) || ~ismatrix(predictions) || isempty(predictions) || ...
        ~isreal(predictions) || any(~isfinite(predictions(:)))
    error('Research:InvalidPredictions', 'Supply a nonempty finite samples-by-models matrix.');
end
labels = mode(predictions, 2);
end
