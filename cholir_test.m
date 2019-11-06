% CHOL_TEST tests the loss of positivity of a SPD matrix by conversion to
% fp16.

clear all; close all;
rng(1);
% Input parameters
cn = 2; % diagonal perturbation constant
fp.format = 'h'; % low precision format to be considered
pflag = 0; % a plot of eignevalues

index = ssget;
indlist = find(index.isReal == 1 & index.numerical_symmetry == 1 & ...
    index.posdef == 1 & index.nrows >= 300 & index.nrows <= 500 & ...
    index.nrows == index.ncols);
[nlist,i] = sort(index.nrows(indlist)) ;
indlist   = indlist(i);
nn = length(indlist);
eval_ctest = zeros(nn,9);
mn = zeros(nn,1);
nlist = nlist';
eval_ctest(:,1) = nlist;
ctest = zeros(nn,1);

fid1 = fopen('cholir_test.txt','w');
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
for j = 1:nn
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);
    Problem = ssget(indlist(j));
    A = full(Problem.A);
    A1 = A;
    mu = 0.1*xmax;
    n = length(A);
    E = mu*cn*u*eye(n);
    
    AbsA = abs(A1);
    mel(j,1) = max(max(AbsA));
    mel(j,2) = min(AbsA(AbsA>0));
    
    % Cholesky factorization test
    [A2,D] = spd_diag_scale(A,0);
    A = mu*A2;
    mn(j,1) = norm(A,inf);
    eval_ctest(j,2) = cond(A);
    
    % single precision Cholesky
    us = float_params('single');
    As = single((A1+(cn*us*diag(diag(A1)))));
    Rs = chol(As);
    Rs = double(Rs);
    eval_ctest(j,3) = cond(Rs\(Rs'\A1));
    clear B;
    
    B1 = A+E;
    Bc = chop(B1,fp);
    [R,flag] = chol_lp(Bc,'h');
    M = mu*(inv(D)*(R'*R)*inv(D));
    eval_ctest(j,4) = cond(A1);
    eval_ctest(j,5) = cond(Bc);
    eval_ctest(j,6) = n/(max([min(eig(A2));(cn*u)]));
    eval_ctest(j,7) = max(eig(A2));
    B1t = mu*(D*(R\(R'\(D*A1))));
    eval_ctest(j,8) = cond(B1t);
    %     B2t = (R\(R'\A))/mu;
    %     eval_ctest(j,8) = cond(B2t,inf);
    
    if ((j == 11 || j == 4) && pflag == 1)
        scatter([1:1:length(B1t)]',sort(eig(B1t),'descend'),250,'d');
        set(gca, 'YScale', 'log')
        set(gca,'FontSize',40)
        ylabel('eigenvalues')
        xlabel('eigenvalue index')
        if j == 11
            [~, objh] = legend('Matrix 11');
        elseif j == 4
            [~, objh] = legend('Matrix 4');
        end
        objhl = findobj(objh, 'type', 'patch'); %// objects of legend of type line
        set(objhl, 'Markersize', 30); %// set marker size as desired
    end
    
    %% GMRES-IR Test with Cholesky of H+c\uhI as preconditioner
    b = randn(nlist(j,1),1);
    scale.flag = 1; scale.type = 'p';
    scale.theta = 0.1; scale.pert = cn;
    scale.precon = 'l';maxit = 10;figs = 0;
    
    % Left preconditioned GMRES based iterative refinement
    %%%(half,single,double)
    [~,irits{1,1}(j,1),gmresits{1,1}(j,1)] = gmresir3(A1,b,1,1,2,maxit,...
        1e-2,scale,figs);
    %%%(half,double,quad)
    [~,irits{2,1}(j,1),gmresits{2,1}(j,1)] = gmresir3(A1,b,1,2,4,maxit,...
        1e-4,scale,figs);
    
    % Preconditioned Conjugate Gradient based iterative refinement
    clear scale;
    scale.flag = 1; scale.theta = 0.1;
    scale.pert = cn;
    %%%(half,single,double)
    [~,irits{1,2}(j,1),gmresits{1,2}(j,1)] = cgir3(A1,b,1,1,2,maxit,...
        1e-2,scale,figs);
    %%%(half,double,quad)
    [~,irits{2,2}(j,1),gmresits{2,2}(j,1)] = cgir3(A1,b,1,2,4,maxit,...
        1e-4,scale,figs);
    
    %% GMRES-IR with Cholesky of A+c*\uh*I as preconditioner
    % compute the preconditioner
    Amax = max(diag(A1));
    And = A1 + (cn*Amax*u*eye(length(A1))); 
    [And1,D1] = spd_diag_scale(And);
    R = chol_lp((mu*And1),'h');
    clear scale;
    % parameters to be used in GMRES-IR
    scale.type = 'g';
    scale.flag = 0; scale.precon = 'l';
    scale.cluf = 1;
    scale.L = (1/sqrt(mu))*diag(1./diag(D1))*R';
    scale.U = (1/sqrt(mu))*R*diag(1./diag(D1));
    scale.P = eye(length(scale.L));
    eval_ctest(j,9) = cond((scale.U)\((scale.L)\(A1)));
    
    % Left preconditioned GMRES based iterative refinement
    %%%(half,single,double)
    [~,irits{1,3}(j,1),gmresits{1,3}(j,1)] = gmresir3(A1,b,1,1,2,maxit,...
        1e-2,scale,1);
    %%%(half,double,quad)
    [~,irits{2,3}(j,1),gmresits{2,3}(j,1)] = gmresir3(A1,b,1,2,4,maxit,...
        1e-4,scale,figs);
    
    % Preconditioned Conjugate Gradient based iterative refinement
    % with (A+c*uh*amax*I) preconditioner
    %%%(half,single,double)
    [~,irits{1,4}(j,1),gmresits{1,4}(j,1)] = cgir3(A1,b,1,1,2,maxit,...
        1e-2,scale,figs);
    %%%(half,double,quad)
    [~,irits{2,4}(j,1),gmresits{2,4}(j,1)] = cgir3(A1,b,1,2,4,maxit,...
        1e-4,scale,figs);
    
    %% GMRES-IR test with (half,double,double) and (single,double,double) 
    clear scale
    scale.pert = cn;
    scale.flag = 1; scale.type = 'p';
    scale.theta = 0.1; 
    scale.precon = 'l';
    
    % Left preconditioned GMRES based iterative refinement
    %%%(half,double,double)
    [~,irits{1,5}(j,1),gmresits{1,5}(j,1)] = gmresir3(A1,b,1,2,2,maxit,...
        1e-2,scale,figs);
    clear scale;
    %%%(single,double,double)
    scale.flag = 0; scale.precon = 'l';
    scale.pert = cn;
    [~,irits{2,5}(j,1),gmresits{2,5}(j,1)] = gmresir3(A1,b,3,2,2,maxit,...
        1e-4,scale,figs);
    clear scale
    
    %% CG-IR test with (half,double,double) and (single,double,double) 
    % Preconditioned Conjugate Gradient based iterative refinement
    scale.flag = 1; scale.type = 'p';
    scale.theta = 0.1; scale.pert = cn;
    %%%(half,double,double)
    [~,irits{1,6}(j,1),gmresits{1,6}(j,1)] = cgir3(A1,b,1,2,2,maxit,...
        1e-2,scale,figs);
    clear scale
    %%%(single,double,double)
    scale.flag = 0;scale.pert = cn;
    [~,irits{2,6}(j,1),gmresits{2,6}(j,1)] = cgir3(A1,b,3,2,2,maxit,...
        1e-4,scale,figs);
end

% print matrix properties
for j=1:nn
    mi = indlist(j);
    fprintf(fid1,'%d & %s & %d & %6.2e & %6.2e & %6.2e\\\\\n',...
        j,index.Name{mi,1},eval_ctest(j,1),eval_ctest(j,4),...
        mel(j,1),mel(j,2));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');


% for j=1:nn
%     fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e & %6.2e\\\\\n',...
%         j,eval_ctest(j,2),eval_ctest(j,3),eval_ctest(j,6),...
%         eval_ctest(j,8),eval_ctest(j,9));
% end
% 
% fprintf(fid1,'\n'); fprintf(fid1,'\n');


for j=1:nn
    fprintf(fid1,'%d& %6.2e & %6.2e & %6.2e& %6.2e\\\\\n',...
        j,eval_ctest(j,2),eval_ctest(j,8),eval_ctest(j,9),eval_ctest(j,3));
end

% for j=1:nn
%     fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e & %6.2e\\\\\n',...
%         j,eval_ctest(j,4),eval_ctest(j,5),eval_ctest(j,6),...
%         eval_ctest(j,7),eval_ctest(j,8));
% end

% creating a text file to print the GMRES and CG iteration table
fprintf(fid1,'\n'); fprintf(fid1,'\n');
for i = 1:nn
    %
    t11  =  gmresits{1,1}(i,1); t22 = irits{1,1}(i,1)-1;
    t33  =  gmresits{2,1}(i,1); t44 = irits{2,1}(i,1)-1;
    %
    t1  =  gmresits{1,2}(i,1); t2 = irits{1,2}(i,1)-1;
    t3  =  gmresits{2,2}(i,1); t4 = irits{2,2}(i,1)-1;
    %
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d)\\\\ \n',i,...
        t11,t22,t1,t2,t33,t44,t3,t4);
end

% creating a text file to print the GMRES and CG iteration for 
% A + c*\uh*amax*I preconditioner 
fprintf(fid1,'\n'); fprintf(fid1,'\n');
for i = 1:nn
    %
    t1a  =  gmresits{1,3}(i,1); t2a = irits{1,3}(i,1)-1;
    t3a  =  gmresits{2,3}(i,1); t4a = irits{2,3}(i,1)-1;
    %
    t1b  =  gmresits{1,4}(i,1); t2b = irits{1,4}(i,1)-1;
    t3b  =  gmresits{2,4}(i,1); t4b = irits{2,4}(i,1)-1;
    %
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d)\\\\ \n',i,...
        t1a,t2a,t1b,t2b,t3a,t4a,t3b,t4b);
end

% creating a text file to print the GMRES and CG iteration for 
% two precision IR
fprintf(fid1,'\n'); fprintf(fid1,'\n');
for i = 1:nn
    %
    t1a  =  gmresits{1,5}(i,1); t2a = irits{1,5}(i,1)-1;
    t3a  =  gmresits{2,5}(i,1); t4a = irits{2,5}(i,1)-1;
    %
    t1b  =  gmresits{1,6}(i,1); t2b = irits{1,6}(i,1)-1;
    t3b  =  gmresits{2,6}(i,1); t4b = irits{2,6}(i,1)-1;
    %
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d)\\\\ \n',i,...
        t1a,t2a,t1b,t2b,t3a,t4a,t3b,t4b);
end

% creating a text file to print the GMRES and CG iteration for 
% tensor core Cholesky preconditioner 
fprintf(fid1,'\n'); fprintf(fid1,'\n');
for i = 1:nn
    %
    t1a  =  gmresits{1,5}(i,1); t2a = irits{1,5}(i,1)-1;
    t3a  =  gmresits{2,5}(i,1); t4a = irits{2,5}(i,1)-1;
    %
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) \\\\ \n',i,...
        t1a,t2a,t3a,t4a);
end

fclose(fid1);

