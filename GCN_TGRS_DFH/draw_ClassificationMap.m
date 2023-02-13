clc;
clear all;
close;

%draw ground truth
TR = double(imread('IndianTR123_temp123.tif'));
TE = double(imread('IndianTE123_temp123.tif'));
[m, n] = size(TR);

GT = TR + TE;
GT2d = hyperConvert2d(GT);
GT2d(GT2d==0)=17;
CM = giveColorCM_HH(GT2d,m,n);
figure;
imshow(uint8(CM));

% draw predicted classificaiton map
load features.mat;

TE2d = hyperConvert2d(TE);
x = find(TE2d > 0);
Pred_TE = zeros(size(TE2d));
[~, ind] = max(features', [], 1);
Pred_TE(:, x) = ind;
Pred_TE3d = hyperConvert3d(Pred_TE, m, n);

Pred_CM = TR + Pred_TE3d;
Pred_CM(Pred_CM==0)=17;
Pred_CM = giveColorCM_HH(Pred_CM,m,n);
figure;
imshow(uint8(Pred_CM));

