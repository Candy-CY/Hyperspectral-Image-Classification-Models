function [indexes]=train_test_random_new(y,n)
% function to ramdonly select training samples and testing samples from the
% whole set of ground truth.
K = max(y);
% generate the  training set
indexes = [];
for i = 1:K
    index1 = find(y == i);
    per_index1 = randperm(length(index1));        
    if length(index1)>60
        indexes = [indexes ;index1(per_index1(1:n(i)))'];
    else
        indexes = [indexes ;index1(per_index1(1:n(i)))'];
    end
end
indexes = indexes(:);


