clc;
clear all;
close all;
X = loadMNISTImages('train-images.idx3-ubyte');
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma);

[U, S] = pca(X_norm);

a=zeros(10000,1);
U=[U a];

K = 100;
Y = loadMNISTImages('t10k-images.idx3-ubyte');

mu = mean(Y);
Y_norm = bsxfun(@minus, Y, mu);

sigma = std(Y);
Y_norm = bsxfun(@rdivide, Y_norm, sigma);

Z = projectData(Y_norm, U, K);
Y_rec  = recoverData(Z, U, K);

D = abs(Y_norm-Y_rec).^2;
MSE = sum(D(:))/numel(Y);

psnr_value=psnr(Y_rec,Y_norm);
