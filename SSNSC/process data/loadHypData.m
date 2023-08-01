function [data,gt, sz, no_lines, no_rows, no_bands, imageName] = loadHypData(dataType)

    disp('LOADING DATA')

%% Load data

if dataType == 1
    load('Salinas_corrected.mat');
    imageName = 'Salinas_corrected';
    data = salinas_corrected;
    clear salinas_corrected;
    load('Salinas_gt.mat');
    gt = double(salinas_gt);
    sz = size(data);
    clear salinas_gt;
    
elseif dataType == 2 % pavia
    load('PaviaU.mat');
    imageName = 'PaviaU';
    data = paviaU;
    clear paviaU;
    
    load('PaviaU_gt.mat');
    gt = double(paviaU_gt);
    sz = size(data);
    clear paviaU_gt;
    
    
elseif dataType == 3 % KSC
    load('KSC.mat');
    imageName = 'KSC';
    data = KSC;
    clear KSC;
    
    load('KSC_gt.mat');
    gt = double(KSC_gt);
    sz = size(data);
    clear KSC_gt;
    
elseif dataType == 4 % pines
    load('Indian_pines_corrected.mat');
    imageName = 'Indian_pines';
    data = indian_pines_corrected;
    clear indian_pines_corrected;
    
    load('Indian_pines_gt.mat');
    gt = double(indian_pines_gt);
    sz = size(data);
    clear indian_pines_gt;
    
elseif dataType == 5 % pines
    load('Pavia.mat');
    imageName = 'PaviaC';
    data = pavia;
    clear pavia;
    
    load('Pavia_gt.mat');
    gt = double(pavia_gt);
    sz = size(data);
    clear pavia_gt;    
elseif dataType == 6 % pines 8 claases
    imageName = 'reducesIndian_pines';
    load('reducesIndian_pines_corrected.mat');
    data = indian_pines_corrected;
    clear indian_pines_corrected;
    
    gt = double(reducedIndian_pines_gt);
    sz = size(data);
    clear reducedIndian_pines_gt;    
    
elseif dataType == 7 % pines
    load('PaviaSub.mat');
    imageName = 'PaviaC';
    data = pavia;
    clear pavia;
    
    load('PaviaSub_gt.mat');
    gt = double(pavia_gt);
    sz = size(data);
    clear pavia_gt; 
elseif dataType == 8 % pines
    load('Botswana.mat');
    imageName = 'Botswana';
    data = Botswana;
    clear Botswana;
    
    load('Botswana_gt.mat');
    gt = double(Botswana_gt);
    sz = size(data);
    clear pavia_gt; 
end
%     imagesc(gt);

[no_lines, no_rows, no_bands] = size(data);

