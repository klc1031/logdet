function [A0, N0,sigma] = geneMatrix(m, p)

rng(2,'twister'); % initialize the random number generator to make the results in this example repeatable.

U= rand(m,p) + 1i*rand(m,p);
%U =[1+2i,2-4i;2-1i,-1+3i;-1+3i,3-1i;1-2i,4-2i];
U_r=orth(U);
sigma=10*rand(1,p);
%sigma=[1;2];
sigma=sort(sigma,'descend');
%sigma=[1000,500,100/3,25,20,50/3,100/7,25/2]
Sigma=diag(sigma);
A0=U_r*Sigma*U_r';
N = p:-1:1;
N0=diag(N);
