function [trainIndexes,testIndexes] = determineTrainAndTestIndices(fixTrainingSize, no_classes, allLabeledData,numberOfInstancesOfEachClass)
disp('Determining random train and test indices')
trainingDataSizeForEachClass = [];
for i =1:no_classes
    trainingDataSizeForEachClass = [trainingDataSizeForEachClass fixTrainingSize];
end
indexes = train_test_random_new(allLabeledData(2,:),trainingDataSizeForEachClass);

% training samples indees
trainIndexes = allLabeledData(:,indexes);

% test samples indees
testIndexes = allLabeledData;
testIndexes(:,indexes) = [];
