function target = prepareTarget(Y)
%PREPARETARGET One-hot columns follow sorted unique training labels.
labels = numericLabels(Y);
classes = unique(labels);
[~, indices] = ismember(labels, classes);
target = zeros(numel(labels), numel(classes));
target(sub2ind(size(target), (1:numel(labels)).', indices)) = 1;
end
