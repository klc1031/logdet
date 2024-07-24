clc
clear
close all

%% For Matlab solver for Riemannian optimization on the Stiefel manifold
%  min -1/2 tr(Jp U'*Jn*A*U*N)
%  s.t.   U'*U = I2p


opts.record = 0;
opts.mxitr  = 3000;
opts.xtol = 1e-6;
opts.gtol = 1e-6;
opts.ftol = 1e-12;
opts.mm = 2;  % backward integer for nonmonotone line search
opts.nt = 5;
opts.alpha = 1.4;
opts.lambda = 0.2;  % factor for decreasing the step size in the backtracking line search
opts.delta = 1e-4;  % parameters for cocntrol the linear approximation in line search

W=[60,80,100,200,400,600,800,1000];
P=[8, 8,8,8,  8,  8,  8,  8];

for i= 1
for j = 1:10

    n=W(i);p=P(i);
    I=eye(p);
    I_n=eye(n);

  
    A=hilb(n);
    eigvalue_A=eig(A);


    %% real form

    N=diag(2*p:-2:1);
    U1 = randn(n,p)+1i*randn(n,p);
    U0 = orth(U1);

    %% Algorithm 4.1--自编程序Riemannian Trust-region method-modified trust region problem----
    % 加预处理的 Trust-region

    tic;
    [U_4,out_RTR]= det_Our_RTR(U0, @det_fun_singular, opts, A,N,I);
    %----output-------
    time_Trust_our=toc;
    time_Trust_our11(j)=time_Trust_our;
    time_Trust_our1=mean(time_Trust_our11);
    itr_Trust_our=out_RTR.itr;
    itr_Trust_our11(j)=itr_Trust_our;
    itr_Trust_our1=mean(itr_Trust_our11);
    grad_Trust_our1=out_RTR.Gradu;
    F_Trust_our1=out_RTR.fval;
    TT1=out_RTR.TT;
    FF1=out_RTR.FF;
    %error1=out_RTR.error;
    diff1=norm(U_4'*U_4-I,"fro");
    %RMSE_Trust_our1=out_RTR.RMSE;
    T1=U_4'*A*U_4;
    lam11=out_RTR.error;
    %laa1(j,:)=lam11;
    %LA1=mean(laa1,1);
    La1=lam11(itr_Trust_our+1);
    lam111(j)=La1;
    lam1=mean(lam111);



    %% Algorithm 4.1--自编程序Riemannian Trust-region method-modified trust region problem----
    % 加预处理的 Trust-region

    %opts.gtol = 1e-6;
    tic;
    [U_8,out_tRTR]= tr_Our_RTR(U0, @tr_fun_singular, opts, A,N);
    %----output-------
    time_Trust_our21=toc;
    time_Trust_our22(j)=time_Trust_our21;
    time_Trust_our2=mean(time_Trust_our22);
    itr_Trust_our21=out_tRTR.itr;
    itr_Trust_our22(j)=itr_Trust_our21;
    itr_Trust_our2=mean(itr_Trust_our22);
    grad_Trust_our2=out_tRTR.Gradu;
    F_Trust_our2=out_tRTR.fval;
    TT2=out_tRTR.TT;
    FF2=out_tRTR.FF;
    lam22=out_tRTR.error;
    La2=lam22(itr_Trust_our21+1);
    lam222(j)=La2;
    lam2=mean(lam222);

    diff2=norm(U_8'*U_8-I,"fro");
    T2=U_8'*A*U_8;


    %% print numerical results
    fprintf('%6s\t  %4s\t %4s\t %4s\t %4s\t %4s\t \n', ...
        'Method',' CPU','IT','lamda','  Fvalue','Grad');
    fprintf('%4s\t  %1.3f\t %1.3f\t %4.2e\t %4.2e\t %4.2e\n', ...
        'ld RTR', time_Trust_our1,itr_Trust_our1,lam1, F_Trust_our1,grad_Trust_our1);
    fprintf('%4s\t  %1.3f\t %1.3f\t %4.2e\t %4.2e\t %4.2e\n', ...
        'tr RTR',  time_Trust_our2,itr_Trust_our2,lam2, F_Trust_our2,grad_Trust_our2);
    fprintf('---------------------------------------------------------------------------------------- \n');

    %% figure

    figure (j)
    hold on

    if itr_Trust_our1<100
        d3=plot(lam11,  'co-','Markersize',6,'LineWidth',1);
    else
        d3=plot(lam11(1:100),  'co-','Markersize',6,'LineWidth',1);
    end

    if itr_Trust_our2<100
        d6=plot(lam22,  'r*-','Markersize',6,'LineWidth',1);
    else
        d6=plot(lam22(1:100),  'r*-','Markersize',6,'LineWidth',1);
    end



    xlabel('Iteration','FontSize',15);
    ylabel('$\|R\|_F$','Interpreter','latex','fontsize',10)
    legend('log-det RTR','tr RTR')%,'SSI''log-det GBB','log-det RCG','det-RTR',,'det-nt','tr-GBB','tr-RCG','tr-RTR',
    set(gca,'LineWidth',1)
    set(gca,'FontSize',15)
    title(['n=',num2str(n)],'fontsize',14)%'n=',num2str(n),',p=',num2str(p)
    hold off
    box on


end
end