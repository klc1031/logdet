function [E_H,F_H]=tr_Hessian_expresion(U,Ubot,E,F,A,N)
%  Compute the Hessian Hessf(X) of the objective function acts on
%  (xi,eta)=(U*E+U_bot*F,V*M+V_bot*N)  as
%  Hess f(U,V)[(xi,eta)]=(U*E_H+U_bot*F_H, V*M_H+V_bot*N_H)

xi=U*E+Ubot*F;
aa1=U'*A*U;
a1=0.5*(aa1*N+N*aa1);
a=-2*U'*A*xi*N+2*U'*xi*a1;
E_H=0.5*(a-a');
F_H=-2*Ubot'*A*xi*N+2*Ubot'*xi*a1;


end