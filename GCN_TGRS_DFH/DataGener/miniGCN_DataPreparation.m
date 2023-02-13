clc;
clear;
close all;

I = double(imread('19920612_AVIRIS_IndianPine_Site3.tif'));
I = I(:,:,[1:103, 109:149,164:219]);
[m, n, z] = size(I);

TR = double(imread('IndianTR123_temp123.tif'));
TE = double(imread('IndianTE123_temp123.tif'));

I2d = hyperConvert2d(I);
for i = 1 : z
    I2d(i,:) = mat2gray(I2d(i,:));
end
TR2d = hyperConvert2d(TR);
TE2d = hyperConvert2d(TE);

TR_sample = I2d(:,TR2d>0);
TE_sample = I2d(:,TE2d>0);

TR_temp = TR2d(:,TR2d>0);
TE_temp = TE2d(:,TE2d>0);

X = [TR_sample,TE_sample];
Y = [TR_temp, TE_temp];

K = 10;
si = 1;
Train_W = creatLap(TR_sample, K, si);
Train_D = (sum(Train_W, 2)).^(-1/2);
Train_D = diag(Train_D);
L_temp = Train_W * Train_D;
Train_L = Train_D * L_temp;
Train_L = Train_L + eye(size(Train_L));

Test_W = creatLap(TE_sample, K, si); 
Test_D = (sum(Test_W, 2)).^(-1/2);
Test_D = diag(Test_D);
L_temp = Test_W * Test_D;
Test_L = Test_D * L_temp;
Test_L = Test_L + eye(size(Test_L));

Train_X = TR_sample';
Test_X = TE_sample';
TrLabel = TR_temp';
TeLabel = TE_temp';

%% Please replace the following route with your own one
save('D:\Python_Project\GCN\HSI_GCN/Train_X.mat','Train_X');
save('D:\Python_Project\GCN\HSI_GCN/Test_X.mat','Test_X');
save('D:\Python_Project\GCN\HSI_GCN/TrLabel.mat','TrLabel');
save('D:\Python_Project\GCN\HSI_GCN/TeLabel.mat','TeLabel');
save('D:\Python_Project\GCN\HSI_GCN/Train_L.mat','Train_L');
save('D:\Python_Project\GCN\HSI_GCN/Test_L.mat','Test_L');
