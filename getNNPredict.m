function labels = getNNPredict(net, X)
%GETNNPREDICT Preserve labels such as 0, -1 or 10 rather than output indices.
if ~isstruct(net.userdata) || ~isfield(net.userdata, 'classLabels')
    error('Research:MissingClassLabels', 'Retrain this network or restore its training classLabels metadata.');
end
labels = decodeClassScores(net(X.'), net.userdata.classLabels);
end
