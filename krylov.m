function [Q,T,itr_k,lam3]=krylov(U0,A,N,I,opts)
%Q0=rand(4,2);
[~,p]=size(U0);
Q = U0;
q=1;
%I=eye(2);
%la1=sort(eig(A), 1, 'descend');
%la=sum(log(real(1+la1(1:p))));
%prev_eigenvalue = eig(A);
%F1=-real(log(det(I+Q'*A*Q*N)));
%lt=sum(log(real(1+eig(Q'*A*Q))));
%f1=(la-lt)/la;
%F=[];
%F=[F,F1];
%f2=[];
%f2=[f2,f1];
mxitr=opts.mxitr;
%eigvalue_A=eig(A);
%sigma1=sort(eigvalue_A,'descend');
%sigma=sigma1(1:p);
%Sigma=diag(sigma);
lam=norm(A*U0-U0*(U0'*A*U0),"fro");
lam3=[];
lam3=[lam3,lam];
%N=diag([2,1])
for itr=1:mxitr
    Q_old=Q;
    Z=A^q*Q_old;
    [Q,~]=qr(Z,0);
    T=Q'*A*Q;
    lam=norm(A*Q-Q*T,"fro");
    lam3=[lam3,lam];
    %f=-real(log(det(I+T*N)));
    %eigenvalue = max(diag(T));
    %lt=sum(log(real(1+eig(T))));
    %ff=(la-lt)/la;
    %f2=[f2,ff];
    % 计算主特征值的变化率
    %eigenvalue_change = abs(eigenvalue - prev_eigenvalue);

    % 更新上一次的主特征值
    %prev_eigenvalue = eigenvalue;

    % 检查变化率是否小于阈值
    if lam <= 1e-6
        break;
    end
    %F=[F,f];
    

end
%Q;
%T;
itr_k=itr;
%plot(F)