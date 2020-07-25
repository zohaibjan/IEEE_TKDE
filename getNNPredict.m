function predict = getNNPredict(net,X)
    x = X';
    y = net(x);
    predict = vec2ind(y)';
end