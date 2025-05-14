function [allLabeledData, GTVector, dataCascade, no_classes, numberOfInstancesOfEachClass] = getLabeledDataAndTheirGTs(gt,img)

sz = size(img);
dataCascade = reshape(img, [prod(sz(1:2)) sz(3)])';
GTVector =  [reshape(gt,[sz(1)*sz(2) 1])]';

for i = 1:max(GTVector)
    numberOfInstancesOfEachClass(i) = sum(GTVector==i);
end
% numberOfInstancesOfEachClass
all = [1:sz(1)*sz(2)];
allLabeledData = [all;GTVector];
allLabeledData(:,allLabeledData(2,:)==0) = [];
no_classes=max(allLabeledData(2,:));
% imagesc(reshape(gt,[sz(1) sz(2)]));
% axis image
end
