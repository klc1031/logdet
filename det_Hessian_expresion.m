function [E_H,F_H]=det_Hessian_expresion(U,Ubot,E,F,A,N,I)
%  Compute the Hessian Hessf(X) of the objective function acts on
%  (xi,eta)=(U*E+U_bot*F,V*M+V_bot*N)  as
%  Hess f(U,V)[(xi,eta)]=(U*E_H+U_bot*F_H, V*M_H+V_bot*N_H)

% a1=U'*A*U*N;
% a2=N*U'*A*U;
% a=linsolve(I+a1,I);
% aa=linsolve(I+a2,I);
% %a=inv(I+a1);
% xi=U*E+Ubot*F;
% eta=xi'*A*U+U'*A*xi;
% b1=a1*(a+aa);
% b=0.5*(b1+b1');
% EH=-U'*A*xi*(N*a+aa*N)+U'*A*U*(N*a*eta*N*a+aa*N*eta*aa*N)+E*b;
% E_H=(EH-EH')/2;
% F_H=-Ubot'*A*xi*(N*a+aa*N)+Ubot'*A*U*(N*a*eta*N*a+aa*N*eta*aa*N)+F*b;
a1=U'*A*U*N;
%a2=N*U'*A*U;
a=linsolve(I+a1,I);
%aa=linsolve(I+a2,I);
%a=inv(I+a1);
xi=U*E+Ubot*F;
eta=xi'*A*U+U'*A*xi;
b1=a1*a;
b=0.5*(b1+b1');
EH=-U'*A*xi*N*a+U'*A*U*N*a*eta*N*a+E*b;
E_H=EH-EH';
F_H=-2*Ubot'*A*xi*N*a+2*Ubot'*A*U*N*a*eta*N*a+2*F*b;
end