function [E,F]=tr_truncated_CG(U,fun,delta,A,N)
%  The truncated CG method for solving the modified trust-region subproblem
[n,p]=size(U);
%---input parameters-------

theta=1;
kappa=0.1;
%-----Initialization---------
E=zeros(p,p); F=zeros(n-p,p);
EFMN_norm=real(sum(dot(E,E,1))+sum(dot(F,F,1)));

[U0, ~] = qr(U);
Ubot = U0(:, p+1:end);  % generate U_{bot}

[~, Gu] = feval(fun, U, A,N);
UG=U'*Gu;
E_g=0.5*(UG-UG');
F_g=Ubot'*Gu;

R_e=E_g; R_f=F_g;
delta_e=-R_e; delta_f=-R_f;

[E_H,F_H]=tr_Hessian_expresion(U,Ubot,delta_e,delta_f,A,N);
R_efmn_norm=real((sum(dot(R_e,R_e,1))+sum(dot(R_f,R_f,1))));
R_Onorm=R_efmn_norm;
%------------------------------
%-----Main iteration-----------------
for itr = 1 : 100
    R_efmn_norm_old=R_efmn_norm;
    H_delta_phi_norm=real(sum(dot(E_H,delta_e,1))+sum(dot(F_H,delta_f,1)));
    
    %------judge negative curvature-----------
    if H_delta_phi_norm<=0
        a=real(sum(dot(delta_e,delta_e,1))+sum(dot(delta_f,delta_f,1)));
        b=real(sum(dot(delta_e,E,1))+sum(dot(delta_f,F,1)));
        c=EFMN_norm;
        tao=(-b+sqrt(b^2+a*(delta^2-c)))/a;
        E=E+tao*delta_e;
        F=F+tao*delta_f; 
        break;
    end
    
    alpha=R_efmn_norm/H_delta_phi_norm;
    E=E+alpha*delta_e; F=F+alpha*delta_f; 
    EFMN_norm=real(sum(dot(E,E,1))+sum(dot(F,F,1)));
    %------judge exceeded trust region-----------
    if EFMN_norm>=delta^2
        a=real(sum(dot(delta_e,delta_e,1))+sum(dot(delta_f,delta_f,1)));
        b=real(sum(dot(delta_e,E,1))+sum(dot(delta_f,F,1)));
        c=EFMN_norm;
        tao=(-b+sqrt(b^2+a*(delta^2-c)))/a;
        E=E+tao*delta_e;
        F=F+tao*delta_f;  
        break;
    end
    R_e=R_e+alpha*E_H; R_f=R_f+alpha*F_H; 
    R_efmn_norm=real((sum(dot(R_e,R_e,1))+sum(dot(R_f,R_f,1))));
    
    beta=R_efmn_norm/R_efmn_norm_old;
    delta_e=-R_e+beta*delta_e;  delta_f=-R_f+beta*delta_f;   
    
    
    %% -----stopping criteria------------
    sqrt_R_norm=sqrt(R_Onorm);
    d=min(sqrt_R_norm^theta, kappa);
    if sqrt(R_efmn_norm)<=max(sqrt_R_norm*d)
        break;
    end
    
    [E_H,F_H]=tr_Hessian_expresion(U,Ubot,delta_e,delta_f,A,N);
    
end
end