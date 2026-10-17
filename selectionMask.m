function selected = selectionMask(weights, threshold)
%SELECTIONMASK Use the same threshold for objective scoring and final output.
if ~isnumeric(weights) || ~isvector(weights) || isempty(weights) || ...
        ~isreal(weights) || any(~isfinite(weights(:))) || ...
        ~isnumeric(threshold) || ~isscalar(threshold) || ~isfinite(threshold)
    error('Research:InvalidSelection', 'Supply finite weights and a finite scalar threshold.');
end
weights = weights(:).';
selected = weights > threshold;
% An empty mask cannot vote. Deterministically retain the largest weight.
if ~any(selected)
    [~, best] = max(weights);
    selected(best) = true;
end
end
