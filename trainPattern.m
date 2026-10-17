function net = trainPattern(X, y)
classLabels = unique(y(:));
x = X.';
y = prepareTarget(y)';
net = patternnet(10);
net.trainParam.showWindow=0;
net = train(net,x,y);
net.userdata.classLabels = classLabels;
% view(net)
% y = net(x);
% classes = vec2ind(y);
end
