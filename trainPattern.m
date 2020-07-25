function net = trainPattern(X, y)
y = prepareTarget(y)';
net = patternnet(10);
net.trainParam.showWindow=0;
net = train(net,x,y);
% view(net)
% y = net(x);
% classes = vec2ind(y);
end
