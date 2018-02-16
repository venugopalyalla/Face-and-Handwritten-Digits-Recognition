clc;
clear all;
close all;
addpath('E:\ASU\Fundamentals of Statistical Learning\Project\faces\faceData')
imgFolder=('E:\ASU\Fundamentals of Statistical Learning\Project\faces\faceData\2002\07\29');
imgSetVector = imageSet(imgFolder,'recursive');
A=[];
for i=1:36
h=imshow(read(imgSetVector,i)); 
A{i} = getimage(h);
end
A=cell2mat(A);
A = im2double(A);
A = rgb2gray(A);

mu = mean(A);
A_norm = bsxfun(@minus, A, mu);

sigma = std(A);
A_norm = bsxfun(@rdivide, A_norm, sigma);

[U, S] = pca(A_norm);

a=zeros(3096,1);
U=[U a];

K = 5;
imgFolder=('E:\ASU\Fundamentals of Statistical Learning\Project\faces\faceData\2002\07\29');
imgSetVector = imageSet(imgFolder,'recursive');
B=[];
for i=1:36
h=imshow(read(imgSetVector,i));
B{i} = getimage(h);
end
B=cell2mat(B);
B = im2double(B);
B = rgb2gray(B);
mu = mean(B);
B_norm = bsxfun(@minus, B, mu);

sigma = std(B);
B_norm = bsxfun(@rdivide, B_norm, sigma);

Z = projectData(B_norm, U, K);
B_rec1  = recoverData(Z, U, K);
B_rec=mat2gray(B_rec1);
D = abs(B_norm-B_rec1).^2;
MSE = sum(D(:))/numel(B);
psnr_value=psnr(B_norm,B_rec1);

B_Show=reshape(B_rec,[86 86 36]);
B_to_disp=reshape(B_Show,[86*86 36]);
figure(1);
display_network(B_to_disp)
figure(2);
BB_Show=reshape(B,[86 86 36]);
BB_to_disp=reshape(BB_Show,[86*86 36]);
display_network(BB_to_disp);
figure(3);
AA_Show=reshape(A,[86 86 36]);
AA_to_disp=reshape(AA_Show,[86*86 36]);
display_network(AA_to_disp);
figure(4);
UU_Show=reshape(U,[86 86 36]);
UU_to_disp=reshape(UU_Show,[86*86 36]);
display_network(UU_to_disp);
