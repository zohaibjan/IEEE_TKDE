function [X]=dataNormalize(D,type)
    if type==1
        max_x=max(D);
        min_x=min(D);
        if any(max_x - min_x ==0)
            X=D;
            return
        end
        X=(D-repmat(min_x,size(D,1),1))./(repmat(max_x,size(D,1),1)-repmat(min_x,size(D,1),1));
    elseif type == 2
        mean_x = mean(D);
        std_x = std(D);
        if any(mean_x == 0)
            X = D;
            return;
        end
          X=(D-repmat(mean_x,size(D,1),1))./(repmat(std_x,size(D,1),1));
    elseif type == 3 %SAMS NORM
        for i = 1:length(D(1,:))
            D(:,i) = mapminmax(D(:,i).').';
            X = D;
        end 
    else
        X=mapminmax(D);
    end
return