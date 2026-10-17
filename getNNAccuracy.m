function acc = getNNAccuracy(net, data)
%GETNNACCURACY Use the training vocabulary even if validation omits a class.
labels = numericLabels(data(:, end));
acc = mean(getNNPredict(net, data(:, 1:end-1)) == labels);
end
