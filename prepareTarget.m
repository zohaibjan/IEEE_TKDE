function [target]=prepareTarget(Y)
    dim=unique(Y);
    id_matrix=eye(length(dim));
    for i=1:size(Y,1)
        for j=1:length(dim)
            if(Y(i)==dim(j))
                target(i,:)=id_matrix(j,:);
            end
        end
    end

end