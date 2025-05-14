function [predLabel, corrMatrix] = ClassificationViaCCA(trainData, fixTrainingSize, cellSpatialTestData, no_classes, testData)
disp('Classification via CCA')
corrMatrix = [];
predLabel = zeros(1,size(testData,2));
for i = 1:size(testData,2)
    correlations = [];
    for k = 1:no_classes
        tempSpatialTestData = cellSpatialTestData{1,i}';
        indx = (k-1)*fixTrainingSize;
        trainDataForEachClass = trainData(:,indx+1:indx+fixTrainingSize);
        [A B r U V] = canoncorr(trainDataForEachClass,tempSpatialTestData);
        warning('off')
        correlations = [correlations r(1)];
    end
    [val predLabel(i) ] = max(correlations);
    corrMatrix = [corrMatrix; correlations];
end
