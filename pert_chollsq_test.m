%PERT_CHOLLSQ_TEST Compares the two shifting strategies for rectangular
%matrices from Suitesparse Matrix collection.

clear all; close all;
rng(1);
% Input parameters
fp.format = 'h'; % low precision format to be considered
chop([],fp); theta = 0.1;
%
index = ssget;
indlist = find(index.isReal == 1 & ...
    index.nrows >= 20 & index.nrows <= 2000 & ...
    index.ncols < 400 & ...
    index.sprank == index.ncols);
[nlist,i] = sort(index.nrows(indlist)) ;
indlist   = indlist(i);
nn = length(indlist);
eval_ctest = zeros(nn,8);
mn = zeros(nn,1);
nlist = nlist';
eval_ctest(:,1) = nlist;
ctest = zeros(nn,1);

fid1 = fopen('pert_chollsq_test.txt','w');
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
a = 0;
for j=1:nn
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);
    Problem = ssget(indlist(j));
    BB = full(Problem.A);

    
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
        D = diag(1./vecnorm(BB));
        mu = theta*xmax;
 
        BB1 = chop((sqrt(mu)*BB*D));
        Ac = hgemm(BB1',BB1,'h');
        As = single(BB')*single(BB);
        flag1 = 0; flag2 = 0; c = 1; n = length(Ac);
        [~,cs1] = chol_lp(Ac); [~,cs2] = chol(As); 
        if (cs1 ~= 0) || (cs2 ~= 0) 
            I = eye(n);
            while flag1 == 0 || flag2 == 0
                if flag1 == 0 && cs1 == 1
                    Af = (1/mu)*(D\diag(diag(Ac))/D);
                    afmax = max(diag(Af));
                    And = Ac + (mu*c*u*afmax*D*D);
                    And1 = (And + And')/2;
                    D1 = D;
%                     [And1,D1] = spd_diag_scale(And);
                    And1 = chop(And1);
                    [~,cs] = chol_lp(And1);
                    if (cs ~= 1)
                        flag1 = 1;
                        rc(a,1) = c;
                    end
                end

                if flag2 == 0 && cs2 ~=0
                    Ad = As + single(c*eps(single(1/2))*diag(diag(As)));
                    [~,cs] = chol(Ad);
                    if (cs == 0)
                        flag2 = 1;
                        rc(a,3) = c;
                    end
                end
                c = c+1;
                if c == 100
                    if flag1 == 0 && cs1 == 1
                        rc(a,1) = Inf;
                    end
                    
                    if flag2 == 0 && cs2 ~=0
                        rc(a,3) = Inf;
                    end
                    break
                end
            end
        end
        
        % quality of the preconditioner
        % fp16 preconditioner
        if cs1 == 0
            And1 = Ac;
            D1 = D;
        end
        [R,~] = chol_lp(And1,'h');
        B1t = mu*(D1*(R\(R'\(D1*(BB'*BB)))));
        rc(a,2) = cond(B1t);
        
        % fp32 preconditioner
        if cs2 == 0
            Ad = As;
        end
        [R,~] = chol(Ad);
        B1t = R\(R'\(BB'*BB));
        rc(a,4) = cond(B1t);
        
    end
end

% print matrix properties
for i = 1:2
    for j=1:a
        
        if i == 1
            if j == a
                fprintf(fid1,'%d\\\\\n',j);
            else
                fprintf(fid1,'%d &',j);
            end
%         elseif i == 2
%             if j == nn
%                 fprintf(fid1,'%6.2e \\\\\n',rc(j,1));
%             else
%                 fprintf(fid1,'%6.2e &',rc(j,1));
%             end
        else
            if j == a
                fprintf(fid1,'%d \\\\\n',rc(j,3));
            else
                fprintf(fid1,'%d &',rc(j,3));
            end
        end
    end
end
% 
% for j=1:a
%     mi = indlist(j);
%     fprintf(fid1,'%d & %d\\\\\n',j...
%                     ,rc(j,3));
% end
fprintf(fid1,'\n'); fprintf(fid1,'\n');
fclose(fid1);
