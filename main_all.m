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

W=[1280,48,112,132,1086,400,100,300,500, 300];
P=[5, 3,5,5,  5,  3,  3,  3,  3,  5];

for i= 2
    %for j = 1:10

    n=W(i);p=P(i);
    I=eye(p);
    I_n=eye(n);

    [A,N,sigma] = geneMatrix(n,p);
    %A = mmread('bcsstm10.mtx');
    %A=diag([7*10^(16), 6*10^(16),6,6,6,6,6,6,6,6]);
    %A_singular = randn(n);
    %A = gallery('binomial', n);
    %A=diag([1,2,3,0,0,0,0,0,0,0])*1e16;
    %inv(A);
    %A=hilb(n);
    %eigvalue_A=eig(A);
    %sigma1=sort(eigvalue_A,'descend');
    %sigma=sigma1(1:p)';
    %Sigma=diag(sigma);
    %A=diag([5,5,5,4.999]);

    %A=diag([2,5,9,2,5]);

    %% real form

    N=diag(p:-1:1);
    U1 = randn(n,p)+1i*randn(n,p);
    U0=orth(U1);
    %U1=ones(n,p);
    %[U0, ~] = qr(U1, 0);
    %T=U0'*A*U0*N;
    %% Algorithm 4.1--自编程序Riemannian Trust-region method-modified trust region problem----
    % 加预处理的 Trust-region

    %opts.gtol = 1e-6;
    tic;
    [U_4,out_RTR]= det_Our_RTR(U0, @det_fun_singular, opts, A,N,I);
    %----output-------
    time_Trust_our1=toc;
    itr_Trust_our1=out_RTR.itr+1;
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
    lam1=lam11(itr_Trust_our1);
    %     lam111(j)=lam1;
    %     La1=mean(lam111);
    %eigvalue1=diag(T1);
    %     eigvalue1=diag(T1);
    %     lam1=norm(T1-Sigma,'fro');
    %TT_RTR_our=[TT_Opt,TT];



    %% Algorithm 4.1--自编程序Riemannian Trust-region method-modified trust region problem----
    % 加预处理的 Trust-region

    %opts.gtol = 1e-6;
    tic;
    [U_8,out_tRTR]= tr_Our_RTR(U0, @tr_fun_singular, opts, A,N);
    %----output-------
    time_Trust_our2=toc;
    itr_Trust_our2=out_tRTR.itr+1;
    grad_Trust_our2=out_tRTR.Gradu;
    F_Trust_our2=out_tRTR.fval;
    TT2=out_tRTR.TT;
    FF2=out_tRTR.FF;
    lam22=out_tRTR.error;
    lam2=lam22(itr_Trust_our2);
    %     lam222(j)=lam2;
    %     La2=mean(lam222);

    diff2=norm(U_8'*U_8-I,"fro");
    %RMSE_Trust_our2=out_RTR.RMSE;
    T2=U_8'*A*U_8;
    %eigvalue2=sort(eig(T2),'descend');
    %eigvalue2=diag(T2);
    %lam2=norm(T2-Sigma,'fro');
    %TT_RTR_our=[TT_Opt,TT];
    %
    tic;
    % for k=1:100
    %     Y=A*U0;
    %     [U_3,r]=qr(Y,0);
    % end
    % T=U_3'*A*U_3;
    %  lam3=norm(diag(T)-Sigma,'fro');
    %  time_k1=toc;
    [U_3,T3,itr_k,lam33] = krylov(U0, A, N, I,opts);
    time_k1=toc;
    itr_k1=itr_k+1;
    %F_k1=F1(itr_k1);
    %error3=f2(itr_k1);
    %diff3=norm(U_3'*U_3-I,"fro");
    %eigvalue3=sort(eig(T3),'descend');
    %eigvalue3=diag(T3);
    %lam3=norm(T3-Sigma,'fro');
    lam3=lam33(itr_k1);
    %     lam333(j)=lam3;
    %     La3=mean(lam333);

%     tic;
%     [V,D]= eig(A);
%     e1=diag(D);
%     eigenvalues = diag(D);
%     [sorted_eigenvalues, indices] = sort(eigenvalues, 'descend');
%     selected_indices = indices(1:p);
%     selected_eigenvalues = sorted_eigenvalues(1:p);
%     selected_vectors = V(:, selected_indices);
%     time_e1=toc;
%     lam4=A*selected_vectors-selected_vectors*(selected_vectors'*A*selected_vectors);
%     lam44=norm(lam4,'fro');

    %     tic;
    %     [U_3,T3,F1,itr_k1,f2] = krylov_det(U0, A, N, I,opts);
    %     time_k1=toc;
    %     F_k1=F1(itr_k1);
    %     error3=f2(itr_k1);
    %     diff3=norm(U_3'*U_3-I,"fro");
    %     %eigvalue3=sort(eig(T3),'descend');
    %     eigvalue3=diag(T3);
    %     lam3=norm(T3-Sigma,'fro');
    %
    %     tic;
    %     [U_5,T4,F2,itr_k2,f4] = krylov_tr(U0, A, N,opts);
    %     time_k2=toc;
    %     F_k2=F2(itr_k2);
    %     error4=f2(itr_k1);
    %     diff4=norm(U_5'*U_5-I,"fro");
    %     %eigvalue4=sort(eig(T4),'descend');
    %     eigvalue4=diag(T4);
    %     lam4=norm(T4-Sigma,'fro');
    %     F2{j}=F;
    %     itr2(j)=itr_k;
    %     time2(j)=time_k;

    %% print numerical results
    fprintf('%6s\t  %4s\t %4s\t %4s\t %4s\t %4s\t \n', ...
        'Method',' CPU','IT','lamda','  Fvalue','Grad');
    fprintf('%4s\t  %1.3f\t %3d\t %4.2e\t %4.2e\t %4.2e\n', ...
        'ld RTR', time_Trust_our1,itr_Trust_our1,lam1, F_Trust_our1,grad_Trust_our1);
    fprintf('%4s\t  %1.3f\t %3d\t %4.2e\t %4.2e\t %4.2e\n', ...
        'tr RTR',  time_Trust_our2,itr_Trust_our2,lam2, F_Trust_our2,grad_Trust_our2);
    %fprintf('%4s\t  %1.3f\t %3d\t %4.2e\t %4.2e\t %4.2e\t %4.2e\n', ...
    %    'ld SSI',  time_k1,itr_k1, F_k1,error3,diff3,lam3);
    %fprintf('%4s\t  %1.3f\t %3d\t %4.2e\t %4.2e\t %4.2e\t %4.2e\n', ...
    %    'tr SSI', time_k2,itr_k2, F_k2,error4,diff4,lam4);
    fprintf('%4s\t  %1.3f\t %3d\t  %4.2e\n', ...
        'SSI', time_k1,itr_k1, lam3);
%     fprintf('%4s\t  %1.3f\t  %4.2e\n', ...
%         'eig', time_e1, lam44);
    fprintf('---------------------------------------------------------------------------------------- \n');

    %% figure
        figure (1)
    
        hold on
    
    
        if itr_Trust_our1<200
            h_Our_RTR1=plot(log10(TT1),  'kx-.','Markersize',6,'LineWidth',1);
        else
            h_Our_RTR1=plot(log10(TT1(1:200)),  'kx-.','Markersize',6,'LineWidth',1);
        end
        if itr_Trust_our2<200
            h_Our_RTR2=plot(log10(TT2),  'b^-.','Markersize',6,'LineWidth',1);
        else
            h_Our_RTR2=plot(log10(TT2(1:200)),  'b^-.','Markersize',6,'LineWidth',1);
        end
    
    
    
        xlabel('Iteration','FontSize',15);
        ylabel('$\|grad~f(U)\|$','Interpreter','latex','fontsize',10)
        %set(gca, 'YScale', 'log')
        yticks(-14:2:4);
        yticklabels({'10^{-14}','10^{-12}','10^{-10}','10^{-8}','10^{-6}','10^{-4}','10^{-2}','0','10^{2}','10^{4}'});
        legend('log-det RTR','tr RTR')%'log-det GBB','log-det RCG','det-nt','tr-GBB','tr-RCG','det-Our-RTR','tr-Our-RTR'
        set(gca,'LineWidth',1)
        set(gca,'FontSize',15)
        %title(['Log-det,','n=',num2str(n),',p=',num2str(p)],'fontsize',14)%
        hold off


   
    figure (2)
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

    if itr_k1<100
        d7=plot(lam33,  'g<-','Markersize',6,'LineWidth',1);
    else
        d7=plot(lam33(1:100),  'g<-','Markersize',6,'LineWidth',1);
    end


    xlabel('Iteration','FontSize',15);
    ylabel('$\|R\|$','Interpreter','latex','fontsize',10)
    %xlim([20, 50]);
    %yticks(-12:2:4);
    %yticklabels({'10^{-12}','10^{-10}','10^{-8}','10^{-6}','10^{-4}','10^{-2}','10^{0}','10^{2}','10^{4}','10^{6}','10^{8}'});
    legend('log-det RTR','tr RTR','SSI')%'log-det GBB','log-det RCG','det-RTR',,'det-nt','tr-GBB','tr-RCG','tr-RTR',
    set(gca,'LineWidth',1)
    set(gca,'FontSize',15)
    %title(['n=',num2str(n)],'fontsize',14)%'n=',num2str(n),',p=',num2str(p)
    hold off
    box on


    %end
end