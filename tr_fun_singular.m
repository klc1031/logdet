function [F,Gu]=tr_fun_singular(U,A,N)
%% -------------------------------------------------------------------------  
F=-trace(U'*A*U*N);
Gu=-2*A*U*N;
