clc;
clear;
close all;

I = double(imread('19920612_AVIRIS_IndianPine_Site3.tif'));
I = I(:,:,[1:103, 109:149,164:219]);
[m, n, z] = size(I);

TR_map = double(imread('IndianTR123_temp123.tif'));
TE_map = double(imread('IndianTE123_temp123.tif'));

I2d = hyperConvert2d(I);
for i = 1 : z
    I2d(i,:) = mat2gray(I2d(i,:));
end
TR2d = hyperConvert2d(TR_map);
TE2d = hyperConvert2d(TE_map);
I = hyperConvert3d(I2d, m, n, z);
[X_train, X_test, X_train_P, X_test_P, Y_train, Y_test] = TR_TE_Generation2d_CNN(I, TR_map, TE_map, 3);
 
save('D:\Python_Project\GCN\HSI_CNN/X_train.mat','X_train');
save('D:\Python_Project\GCN\HSI_CNN/X_test.mat','X_test');
save('D:\Python_Project\GCN\HSI_CNN/Y_train.mat','Y_train');
save('D:\Python_Project\GCN\HSI_CNN/Y_test.mat','Y_test');
