function [U,out]= tr_Our_RTR(U,fun, opts, A,N)
%%  Problem: The turncated complex singular value decomposition problem
%  min  F(U,V)= -Tr(U'AV\Pi))  (U,V) \in St(m,p,C)*St(n,p,C)
%                                U'*U = I_p, where U \in C^{n,p}
%------------------------------------------------------------------------
[n,p]=size(U);
%la1=eig(A);
%la1=sort(eig(A), 1, 'descend');
%la=real(sum(la1(1:p)));
deltabar=2*n*p-p^2;
rho_pie=0.1;
delta=deltabar/8;
eye2p = eye(2*p);
%eigvalue_A=eig(A);
%sigma1=sort(eigvalue_A,'descend');
%sigma=sigma1(1:p);
%Sigma=diag(sigma);
lam2=[];
lam=norm(A*U-U*(U'*A*U),"fro");
lam2=[lam2,lam];
% initialMatrix = rand(n, p);
% matrixHistory = zeros(size(initialMatrix, 1), size(initialMatrix, 2), opts.mxitr);
% matrixHistory(:, :, 1) = U;
%---Used when Cayley transform as Retraction----------
invU = true;
if p < n/2
    invU = false;
end
[Fval,Gu] = feval(fun,U,A,N);  % Function value at X0
FF=[];
FF(1)=real(Fval);
%% main iteration
for itr = 1 : opts.mxitr
    Fval_old=Fval;
    %% 计算Hess特征值的分布
    %HH=hess_xi(n,A,U,N);
    %eig_h=eig(HH)

    % 1、 Output E,F,M,N by solving the modified trust-region subproblem------
    [E,F]=tr_truncated_CG(U,@tr_fun_singular,delta,A,N);
    
    % 2、generate the trial search direction----------
    [U0, ~] = qr(U);
    Ubot = U0(:, p+1:end);  % generate U_{bot}
    xi=U*E+Ubot*F;          % generate xi
    % 3 -----Compute rho_k-------
    %3.1、-----genrate the trial update iteratie X by using the Cayley transformation------
    UG=U'*xi;
    PZu=xi-0.5*U*UG;
    if invU
        WZu = PZu*U' - U*PZu';  Ru = WZu*U;
    else
        Uu=[PZu,U]; Wu=[U,-PZu];
        MU1=Wu'*U; MU2=Wu'*Uu;
    end
    if invU
        U_new = linsolve(eye(n) - 0.5*WZu, U + 0.5*Ru);
    else
        MMu=linsolve(eye2p-0.5*MU2,MU1);
        U_new=U+Uu*MMu;
    end
    
    %3.2、---------compute Eg,Fg,E_H,F_H and rho-------------------------
    UG=U'*Gu;
    %Gradu=Gu-0.5*U*(UG+UG');
    E_g=0.5*(UG-UG');
    %E_g=U'*Gradu;
    F_g=Ubot'*Gu;
    
    [E_H,F_H]=tr_Hessian_expresion(U,Ubot,E,F,A,N);
    %H=U*E_H+Ubot*F_H;
    %eh=sqrt(eig(H'*H));
    [Fval, ~] = feval(fun, U_new,A,N);  % Function value at U_new and V_new
    a=Fval-Fval_old;
    E_t=E_g+0.5*E_H; F_t=F_g+0.5*F_H;
    b=real(sum(dot(E_t,E,1))+sum(dot(F_t,F,1)));
    rho_k=a/b;
    
    %4、 determine the trust-region radious--------------------------
    if rho_k<=0.25
        delta=0.25*delta;
    elseif rho_k>=0.75 && real(sum(dot(xi,xi,1)))-delta^2<1e-8
        delta=min(2*delta,deltabar);
    else
    end
    
    %5、 determine the new update iteratie--------------------------
    if rho_k>rho_pie
        U=U_new;
        %V=V_new;
    else
    end
    
    
    %% -----stopping criteria------------
    [Fval,  Gu] = feval(fun, U, A,N);
    UG=U'*Gu;    Gradu=Gu-0.5*U*(UG+UG');
    Gradu  = sqrt(sum(dot(Gradu, Gradu,1)));
    T=U'*A*U;
    lam=norm(A*U-U*T,"fro");
    lam2=[lam2,lam];
    %lt=sum(real(eig(T)));
    TT(itr)=Gradu;
    FF(itr+1)=real(Fval);
    %ff=(la-lt)/la;
    %f(itr)=ff;    
    if  lam < opts.gtol
        break;
    end


    %matrixHistory(:, :, itr+1) = U;
    
end
% 计算每次迭代生成的矩阵与最后一个矩阵的差异
% for i = 1:itr
%     diffMatrix = matrixHistory(:, :, itr) - matrixHistory(:, :, i);
%     %disp(['Difference between iteration ' num2str(i) ' and the last iteration:']);
%     %disp(diffMatrix);
%     RMSE(i)=norm(diffMatrix,"fro")/sqrt(n*p);
% end
out.Gradu = Gradu;
out.fval = Fval;
out.itr = itr;
out.TT=TT;
out.FF=FF;
out.error=lam2;
%out.RMSE=RMSE;
end