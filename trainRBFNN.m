function network = trainRBFNN(X, y, params , p)
   warning off;
    x = X;
    t = prepareTarget(y)';
    eg = 0.03; % sum-squared error goal
    sc = 1;    % spread constant
    net = newrb(x,t,eg,sc);
    network = net;
end
