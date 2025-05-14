function [HSI_TR, HSI_TE, HSI_TR_P, HSI_TE_P, TR2d, TE2d] = TR_TE_Generation2d_CNN(HSI, TR, TE, batchsize)

num_TR = sum(sum(TR > 0));
num_TE = sum(sum(TE > 0));

HSI = padarray(HSI,[batchsize,batchsize], 'replicate');
TR = padarray(TR,[batchsize,batchsize], 'replicate');
TE = padarray(TE,[batchsize,batchsize], 'replicate');

batch_size = 2 * batchsize + 1;

[m, n, z] = size(HSI);

HSI_TR = zeros(num_TR, batch_size * batch_size * z);
HSI_TE = zeros(num_TE, batch_size * batch_size * z);

HSI_TR_P = zeros(num_TR, z);
HSI_TE_P = zeros(num_TE, z);

TR2d = zeros(num_TR, 1);
TE2d = zeros(num_TE, 1);

k1 = 0;
k2 = 0;
k3 = 0;

for j = (1 + batchsize) : n - batchsize
    for i = (1 + batchsize) : m - batchsize
        if TR(i, j) ~= 0
            k1 = k1 + 1;
            temp_HSI = (reshape(HSI(i - batchsize : i + batchsize, j - batchsize : j + batchsize, :), batch_size * batch_size * z, 1));
            HSI_TR(k1, :) = (temp_HSI);
            HSI_TR_P(k1, :) = HSI(i, j, :);
            TR2d(k1, :) = hyperConvert2d(TR(i, j));
        end
        if TE(i, j) ~=0
            k2 = k2 + 1;
            temp_HSI = (reshape(HSI(i - batchsize : i + batchsize, j - batchsize : j + batchsize, :), batch_size * batch_size * z, 1));
            HSI_TE(k2, :) = (temp_HSI);
            HSI_TE_P(k2, :) = HSI(i, j, :);
            TE2d(k2,:) = hyperConvert2d(TE(i, j));
        end
    end
end

end