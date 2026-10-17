function labels = numericLabels(values)
%NUMERICLABELS Preserve numeric category names instead of category indices.
if isnumeric(values) || islogical(values)
    labels = double(values);
elseif isa(values, 'categorical') || isstring(values) || iscellstr(values)
    labels = str2double(cellstr(values));
else
    error('Research:InvalidLabels', 'Labels must be numeric or numeric category names.');
end
if ~isvector(labels) || isempty(labels) || ~isreal(labels) || any(~isfinite(labels(:)))
    error('Research:InvalidLabels', 'Labels must be a nonempty finite numeric vector.');
end
labels = labels(:);
end
