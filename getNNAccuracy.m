function acc = getNNAccuracy(net,data)
    x = data(:, 1:end-1)';
    t = prepareTarget(data(:,end))';
    
    y = net(x);
    tind = vec2ind(t);
    yind = vec2ind(y);
    acc = mean (yind == tind);
end