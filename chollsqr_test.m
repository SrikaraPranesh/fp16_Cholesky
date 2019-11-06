% CHOLLSQR_TEST tester for fp16 Cholesky based least square
%               solver.

clear all; close all;
rng(1);
% Input parameters
iter_max = [10;10];
gtol = 1e-4;
scale.theta = 0.1;  % diagonal perturbation constant
fp.format = 'h'; % low precision format to be considered

% & index.nrows > index.ncols
index = ssget;
indlist = find(index.isReal == 1 & ...
    index.nrows >= 20 & index.nrows <= 2000 & ...
    index.ncols < 400 & ...
    index.sprank == index.ncols);
[nlist,i] = sort(index.nrows(indlist)) ;
indlist   = indlist(i);
nn = length(indlist);
eval_ctest = zeros(nn,9);
mn = zeros(nn,1);
nlist = nlist';
ctest = zeros(nn,1);

fid1 = fopen('chollsqr_test.txt','w');
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
chop([],fp);

a = 0;
for j=1:nn
    % for j=1:175
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);
    Problem = ssget(indlist(j));
    BB = full(Problem.A);
    scale.pert = 12;
    %
    AbsA = abs(BB);
    mel(j,1) = max(max(AbsA));
    mel(j,2) = min(AbsA(AbsA>0));
    
    
    [d,r(1,1)] = size(BB);
    r(1,2) = rank(BB);
    condBB(j,1) = cond(BB);
    if ((r(1,1) == r(1,2)) && (d > r(1,1)) && (condBB(j,1) < 1e5))
        a = a+1;
        
        act_ind(a,1) = j;
        rows(a,1) = d;
        eval_ctest(a,1) = r(1,1);
        eval_ctest(a,4) = d;
        % column scaling
        A1 = BB'*BB;
        D = diag(1./vecnorm(BB));
        mu = scale.theta*xmax;
        
        
        BB1 = chop((sqrt(mu)*BB*D));
        eval_ctest(a,8) = cond((BB*D));
        A = hgemm(BB1',BB1,'h');
        % perturbation for H
        Eh = mu*scale.pert*u*eye(length(A));
        
        % pertubration for A^{T}A with fp16 factorization
        afmax = max(diag((1/mu)*(D\diag(diag(A))/D)));
        E1 = mu*scale.pert*u*afmax*D*D;
        E1 = (E1 + E1')/2;
        B2 = A+Eh;
        [R2,~] = chol_lp(B2,'h');
        B2t = mu*(D*(R2\(R2'\(D*A1))));
        eval_ctest(a,9) = cond(B2t);
        
        % Cholesky factorization test
        %         [A,D] = spd_diag_scale(A,0);
        %         A = mu*A;
        mn(a,1) = norm(A,inf);
        eval_ctest(a,2) = min(eig(A))/max(eig(A));
        
        % pertubration for A^{T}A with fp32 factorization
        Es = 2*eps(single(1/2))*single(diag(diag(A1)));
        B2 = (single(BB')*single(BB))+single(Es);
        [R2,~] = chol(B2);
        R2 = double(R2);
        B2t = R2\(R2'\A1);
        eval_ctest(a,3) = cond(B2t);
        clear B;
        
        B1 = A+Eh;
        B1 = chop(B1,fp);
        %     eval_ctest(j,4) = min(eig(B1));
        [R,flag] = chol_lp(B1,'h');
        eval_ctest(a,5) = cond(B1);
        eval_ctest(a,6) = cond(BB);
        B1t = mu*(D*(R\(R'\(D*A1))));
        eval_ctest(a,7) = cond(B1t);
        %     B2t = (R\(R'\A))/mu;
        %     eval_ctest(j,8) = cond(B2t,inf);
        
        
        % GMRES-IR Test
        [m,n] = size(BB);
        b = randn(m,1);
        
        % perturbation to diagonally scaled matrix
        % (half,single,double)
        [xsg,its{1,1}(a,1),t_gmres_its{1,1}(a,1)] = ...
            lsq_gmresir3(BB,b,1,1,2,iter_max,1e-2,scale);
        r = double(mp(b)-mp(BB)*mp(xsg));
        nbe{1,1}(a,1) = lsq_be(BB,r,xsg,1);
        nbe{1,1}(a,2) = lsq_be(BB,r,xsg,Inf);
        
        % (half,double,quad)
        [xdg,its{2,1}(a,1),t_gmres_its{2,1}(a,1)] = ...
            lsq_gmresir3(BB,b,1,2,4,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xdg));
        nbe{2,1}(a,1) = lsq_be(BB,r,xdg,1);
        nbe{2,1}(a,2) = lsq_be(BB,r,xdg,Inf);
        
        % Preconditioned Conjugate Gradient based iterative refinement
        % (half,single,double)
        [xsc,its{1,2}(a,1),t_gmres_its{1,2}(a,1)] = ...
            lsq_cgir3(BB,b,1,1,2,iter_max,1e-2,scale);
        r = double(mp(b)-mp(BB)*mp(xsc));
        if norm(xsc)~= 0
            nbe{1,2}(a,1) = lsq_be(BB,r,xsc,1);
            nbe{1,2}(a,2) = lsq_be(BB,r,xsc,Inf);
        else
            nbe{1,2}(a,1) = 1;
            nbe{1,2}(a,2) = 1;
        end
        % (half,double,quad)
        [xdc,its{2,2}(a,1),t_gmres_its{2,2}(a,1)] = ...
            lsq_cgir3(BB,b,1,2,4,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xdc));
        
        if norm(xdc) ~= 0
            nbe{2,2}(a,1) = lsq_be(BB,r,xdc,1);
            nbe{2,2}(a,2) = lsq_be(BB,r,xdc,Inf);
        else
            nbe{2,2}(a,1) = 1;
            nbe{2,2}(a,2) = 1;
        end
        
        % (half,double,double) combintation
        % GMRES
        [xsg,its{1,3}(a,1),t_gmres_its{1,3}(a,1)] = ...
            lsq_gmresir3(BB,b,1,2,2,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xsg));
        nbe{1,3}(a,1) = lsq_be(BB,r,xsg,1);
        nbe{1,3}(a,2) = lsq_be(BB,r,xsg,Inf);
        
        % Preconditioned Conjugate Gradient based iterative refinement
        [xsc,its{2,3}(a,1),t_gmres_its{2,3}(a,1)] = ...
            lsq_cgir3(BB,b,1,2,2,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xsc));
        if norm(xsc)~=0
            nbe{2,3}(a,1) = lsq_be(BB,r,xsc,1);
            nbe{2,3}(a,2) = lsq_be(BB,r,xsc,Inf);
        else
            nbe{2,3}(a,1) = 1;
            nbe{2,3}(a,2) = 1;
        end
        
        % (single,double,double) combintation
        scale.pert = 2;
        % GMRES
        [xsg,its{1,4}(a,1),t_gmres_its{1,4}(a,1)] = ...
            lsq_gmresir3(BB,b,3,2,2,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xsg));
        nbe{1,4}(a,1) = lsq_be(BB,r,xsg,1);
        nbe{1,4}(a,2) = lsq_be(BB,r,xsg,Inf);
        
        % Preconditioned Conjugate Gradient based iterative refinement
        [xsc,its{2,4}(a,1),t_gmres_its{2,4}(a,1)] = ...
            lsq_cgir3(BB,b,3,2,2,iter_max,1e-4,scale);
        r = double(mp(b)-mp(BB)*mp(xsc));
        if norm(xsc)~=0
            nbe{2,4}(a,1) = lsq_be(BB,r,xsc,1);
            nbe{2,4}(a,2) = lsq_be(BB,r,xsc,Inf);
        else
            nbe{2,4}(a,1) = 1;
            nbe{2,4}(a,2) = 1;
        end
        
    end
end

% print matrix properties
for j=1:a
    jj = act_ind(j,1);
    mi = indlist(jj);
    fprintf(fid1,'%d & %s &(%d,%d)& %6.2e & %6.2e & %6.2e\\\\\n',...
        j,index.Name{mi,1},eval_ctest(j,4),eval_ctest(j,1),eval_ctest(j,6),...
        mel(j,1),mel(j,2));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

fprintf(fid1,'Condition numbers and eigenvalues dummy \n');
for j=1:a
    fprintf(fid1,'%d &  %6.2e & %6.2e & %6.2e & %6.2e\\\\\n',...
        j,eval_ctest(j,8),eval_ctest(j,2),eval_ctest(j,7),eval_ctest(j,9));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

fprintf(fid1,'Condition numbers\n');
for j=1:a
    fprintf(fid1,'%d & %6.2e& %6.2e\\\\\n',...
        j,eval_ctest(j,9),eval_ctest(j,3));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');


% creating a text file to print the GMRES iteration table
fprintf(fid1,'\n'); fprintf(fid1,'\n');
fprintf(fid1,'number of GMRES and CG iterations perturb H \n');
for i = 1:a
    
    ta = its{1,1}(i,1); tb = its{1,2}(i,1);
    tc = its{2,1}(i,1); td = its{2,2}(i,1);
    te = its{1,4}(i,1); tf = its{2,4}(i,1);
    
    t2  =  t_gmres_its{1,1}(i,1); t22  =  t_gmres_its{1,2}(i,1);
    t4  =  t_gmres_its{2,1}(i,1); t44  =  t_gmres_its{2,2}(i,1);
    t5  =  t_gmres_its{1,4}(i,1); t55  =  t_gmres_its{2,4}(i,1);
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d)\\\\ \n',...
        i,t2,ta,t22,tb,t4,tc,t44,td,t5,te,t55,tf);
end

% creating a text file to print the GMRES iteration table including (half,double,double)
fprintf(fid1,'\n'); fprintf(fid1,'\n');
fprintf(fid1,'number of GMRES and CG iterations includeing half,double,double combination \n');
for i = 1:a
    
    ta = its{1,1}(i,1); tb = its{1,2}(i,1);
    tc = its{2,1}(i,1); td = its{2,2}(i,1);
    te = its{1,3}(i,1); tf = its{2,3}(i,1);
    
    t2  =  t_gmres_its{1,1}(i,1); t22  =  t_gmres_its{1,2}(i,1);
    t4  =  t_gmres_its{2,1}(i,1); t44  =  t_gmres_its{2,2}(i,1);
    t5  =  t_gmres_its{1,3}(i,1); t55  =  t_gmres_its{2,3}(i,1);
    fprintf(fid1,'%d & %d &(%d) & %d &(%d) & %d &(%d) & %d &(%d)& %d &(%d) & %d &(%d)\\\\ \n',...
        i,t2,ta,t22,tb,t4,tc,t44,td,t5,te,t55,tf);
end


% creating a text file to print the actual normwise backward error
fprintf(fid1,'\n'); fprintf(fid1,'\n');
fprintf(fid1,'Backward error for GMRES-IR\n')
for i = 1:a
    t1 = nbe{1,1}(i,1); t2 = nbe{1,1}(i,2);
    t3 = nbe{2,1}(i,1); t4 = nbe{2,1}(i,2);
    fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e\\\\ \n',i,...
        t1,t2,t3,t4);
end

fprintf(fid1,'\n'); fprintf(fid1,'\n');
fprintf(fid1,'Backward error for CG-IR\n');
for i = 1:a
    t1 = nbe{1,2}(i,1); t2 = nbe{1,2}(i,2);
    t3 = nbe{2,2}(i,1); t4 = nbe{2,2}(i,2);
    fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e\\\\ \n',i,...
        t1,t2,t3,t4);
end

fclose(fid1);

%%%%%%%%%%%%%%%%%%%%%%%
function eta = lsq_be(A,r,x,theta)
% for more details refer to 'Accuracy and Stability of Numerical
% Algorithms -- Nicholas J. Higham', Second Edition, p -- 393.
[m,n] = size(A);
t = (theta^2)*(x'*x);
if theta ~= Inf
    mu = t/1+t;
else
    mu = 1;
end
phi = sqrt(mu)*(norm(r)/norm(x));
D = [A (phi*(eye(m)-(r*pinv(r))))];
smin = min(svd(D));
eta = min([phi smin]);
end

