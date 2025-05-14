function [trainData, trainLabel, testData, testLabel] = determineTrainAndTestData(trainIndexes, testIndexes, dataCascade, gtVector, sz)
disp('Determine train and test data')
trainData = dataCascade(:,trainIndexes(1,:));
trainLabel = gtVector(trainIndexes(1,:));
trainLabelPlot = zeros(sz(1)*sz(2),1);
trainLabelPlot(trainIndexes(1,:)) = gtVector(trainIndexes(1,:));
% imagesc(reshape(trainLabelPlot,[sz(1) sz(2)]))
% colormap(map)

testData = dataCascade(:,testIndexes(1,:));


testLabel = gtVector(testIndexes(1,:));
testLabelPlot = zeros(sz(1)*sz(2),1);
testLabelPlot(testIndexes(1,:)) = gtVector(testIndexes(1,:));
