function [cellSpatialDataCascade] = getSpatialData(dataCascade, sz, trainIndexes, spatialSize, removeTrainingInstancesInNeighborhood)

    %% Get spatial data
    disp('Generating spatial data to be used')
    % plot(dataCascade(:,1))
    if (removeTrainingInstancesInNeighborhood == 1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%   ZERO PADDING TO TRAINING INSTANCE   %%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('    We remove the training samples that are in spatial neighborhood of the test samples')
    disp('    during classification while employing the spectral-spatial methods for fair comparison.')
    dataCascade(:,trainIndexes(1,:)) = zeros([sz(3) size(trainIndexes,2)]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    dataDerivatedNormalized = reshape(dataCascade',[sz(1) sz(2) sz(3)]);
    
    % plot(reshape(dataDerivatedNormalized(1,1,:),[1 204]))
    
    if 5 == spatialSize
    cellSpatialData = generateSpatialData5x5(dataDerivatedNormalized);
    elseif 3 == spatialSize
    cellSpatialData = generateSpatialData(dataDerivatedNormalized);
    elseif 7 == spatialSize
    cellSpatialData = generateSpatialData7x7(dataDerivatedNormalized);
    elseif 9 == spatialSize
    cellSpatialData = generateSpatialData9x9(dataDerivatedNormalized);
    elseif 11 == spatialSize
    cellSpatialData = generateSpatialData11x11(dataDerivatedNormalized);
    elseif 1 == spatialSize
    cellSpatialData = generateSpatialData1x1(dataDerivatedNormalized);
    end
    % plot(cell2mat(cellSpatialData(1,1))')
    cellSpatialDataCascade = reshape(cellSpatialData,[1 sz(1)*sz(2)]);
