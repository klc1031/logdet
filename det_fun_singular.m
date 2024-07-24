function [F,Gu]=det_fun_singular(U,A,N,I)
%% -------------------------------------------------------------------------
%I=eye(2);
UA=U'*A*U;
F=-log(det(I+UA*N)); % objective function
U1=linsolve(I+UA*N,I);
%U2=linsolve(I+N*UA,I);
Gu=-2*A*U*N*U1; % the derivation of  F(U) with respect to U   

