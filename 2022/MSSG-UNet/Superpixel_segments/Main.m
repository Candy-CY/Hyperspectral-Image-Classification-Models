clear; close all;

FLAG=8;
switch(FLAG)
    case 1
        hsi=load('..\..\HyperImage_data\indian\Indian_pines_corrected.mat');hsi=hsi.indian_pines_corrected;
        n_superpixels=[2048,1024,512,256];
    case 2
        hsi=load('..\..\HyperImage_data\paviaU\PaviaU.mat');hsi=hsi.paviaU;
        n_superpixels=[2048,1024,512,256];
    case 3
        hsi=load('..\..\HyperImage_data\Salinas\Salinas_corrected.mat');hsi=hsi.salinas_corrected;
        n_superpixels=[2048,1024,512,256];
    case 4
        hsi=load('..\..\HyperImage_data\KSC\KSC.mat');hsi=hsi.KSC;
        n_superpixels=[2048,1024,512,256];
    case 5
        hsi=load('..\..\HyperImage_data\Houston2013\Houston.mat');hsi=hsi.Houston;
        n_superpixels=[2048,1024,512,256];
    case 6
        hsi=load('..\..\HyperImage_data\HyRANK\Loukia.mat');hsi=hsi.Loukia;
        n_superpixels=[2048,1024,512,256];
    case 7
        hsi=load('..\..\HyperImage_data\Botswana\Botswana.mat');hsi=hsi.Botswana;
        n_superpixels=[2048,1024,512,256];
    case 8
        hsi=load('..\..\HyperImage_data\Houston2018\HoustonU.mat');hsi=hsi.houstonU;
        n_superpixels=[2048,1024,512,256];
        hsi=hsi(:,1:400,:);%
    case 9
        hsi=load('..\..\HyperImage_data\xuzhou\xuzhou.mat');hsi=hsi.xuzhou;
        n_superpixels=[2048,1024,512,256];
    case 10
        hsi=load('..\..\HyperImage_data\WDC\WDC.mat');hsi=hsi.wdc;
        n_superpixels=[2048,1024,512,256];
end

hsi=double(hsi);
[h,w,c]=size(hsi);
hsi=mapminmax( reshape(hsi,[h*w,c])');
hsi=reshape(hsi',[h,w,c]);
hsi = imfilter(hsi, fspecial('gaussian',[5,5]), 'replicate');
hsi=reshape(hsi,[h*w,c])';
pcacomps=pca(hsi);
I=pcacomps(:,[3,2,1])';
I=(mapminmax(I)+1)/2*255;
I=reshape(uint8(I)',[h,w,3]);
for i=1:3
    I(:,:,i)=imadjust(histeq(I(:,:,i))); %%
end
I = imfilter(I, fspecial('unsharp',0.05), 'replicate');
E=uint8(zeros([h,w]));

% fine detail structure
tic; sh=SuperpixelHierarchyMex(I,E,0.0,0.1); toc
segmentmaps=zeros(size(n_superpixels,2),h,w);
for i=1:size(n_superpixels,2)
    GetSuperpixels(sh,n_superpixels(:,i));
    segmentmaps(i,:,:)=sh.label;
end

switch(FLAG)
    case 1
        save segmentmapsindian.mat segmentmaps
    case 2
        save segmentmapspaviau.mat segmentmaps
    case 3
        save segmentmapssalinas.mat segmentmaps
    case 4
        save segmentmapsksc.mat segmentmaps
    case 5
        save segmentmapshst.mat segmentmaps
    case 6
        save segmentmapsloukia.mat segmentmaps
    case 7
        save segmentmapsbot.mat segmentmaps
    case 8
        save segmentmapshstu.mat segmentmaps
    case 9
        save segmentmapsxuzhou.mat segmentmaps
    case 10
        save segmentmapswdc.mat segmentmaps
end

% get whatever you want
GetSuperpixels(sh,n_superpixels(1)); color1 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(2)); color2 = MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(3)); color3= MeanColor(double(I),sh.label);
GetSuperpixels(sh,n_superpixels(4)); color4= MeanColor(double(I),sh.label);
figure,imshow([color1,color2; color3,color4]);
