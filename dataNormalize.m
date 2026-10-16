function X = dataNormalize(D, type)
%dataNormalize Normalize columns; constant columns map to zero in modes 1 and 2.
% Modes: 1 = min/max, 2 = sample z-score, 3 = column-wise mapminmax.
% Other modes preserve the historical row-wise mapminmax behaviour.
if isempty(D)
    X = D;
    return;
end
if type == 1
    offset = min(D, [], 1);
    scale = max(D, [], 1) - offset;
    scale(scale == 0) = 1;
    X = bsxfun(@rdivide, bsxfun(@minus, D, offset), scale);
elseif type == 2
    offset = mean(D, 1);
    scale = std(D, 0, 1);
    scale(scale == 0) = 1;
    X = bsxfun(@rdivide, bsxfun(@minus, D, offset), scale);
elseif type == 3
    X = D;
    for i = 1:size(D, 2)
        X(:, i) = mapminmax(D(:, i).').';
    end
else
    X = mapminmax(D);
end
end
