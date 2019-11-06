%PERT_CHOLIR_TEST Compares the two shifting strategies for square matrices 
%from Suitesparse Matrix collection. 

clear all; close all;
rng(1);
% Input parameters
fp.format = 'h'; % low precision format to be considered
chop([],fp);
%
index = ssget;
indlist = find(index.isReal == 1 & index.numerical_symmetry == 1 & ...
    index.posdef == 1 & index.nrows >= 300 & index.nrows <= 500 & ...
    index.nrows == index.ncols);
[nlist,i] = sort(index.nrows(indlist)) ;
indlist   = indlist(i);
nn = length(indlist);


fid1 = fopen('pert_cholir_test.txt','w');
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
mu = 0.1*xmax;
rc = zeros(nn,4);a = 1; c_spd = zeros(nn,1);
for j=1:nn
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);
    Problem = ssget(indlist(j));
    A = full(Problem.A);
    n = length(A);
    [A2,D] = spd_diag_scale(A);
    amax = max(diag(A));
    Ac = chop(A2);
    flag1 = 0; flag2 = 0; c = 1;
    flag3 = 0;
    [~,cs1] = chol_lp(Ac);
    if (cs1 ~= 0)
        I = eye(n);
        while flag1 == 0 || flag2 == 0 || flag3 == 0
            if flag1 == 0
                And = A + (c*amax*u*I);
                [And1,D1] = spd_diag_scale(And);
                And1 = chop(mu*And1);
                [~,cs] = chol_lp(And1,'h');
                if (cs ~= 1)
                    flag1 = 1;
                    rc(j,1) = c;
                end
            end
            
            if flag2 == 0
                Ad = chop(mu*(A2 + (c*u*I)));
                [~,cs] = chol_lp(Ad,'h');
                if (cs ~= 1)
                    flag2 = 1;
                    rc(j,3) = c;
                end
            end

            % single precision
            if flag3 == 0
                us = float_params('s');
                Ad = single(A + (c*us*diag(diag(A))));
                [~,cs] = chol(Ad);
                if (cs == 0)
                    flag3 = 1;
                    rc(j,4) = c;
                end
            end
            
            c = c+1;
            if c == 100
               if flag1 == 0
                   rc(j,1) = Inf;
               end
               
               if flag2 == 0
                  rc(j,3) = Inf; 
               end
               break
            end
        end
        a = a+1;
    end
    
    
    [H,D] = spd_diag_scale(A);
    H = single(H);
    if min(eig(H)) < 0
        spd_flag = 0; c_spd(j,1) = 0;
        while spd_flag == 0
            H = H + (c_spd(j,1)*eps('single')*eye(n));
            if min(eig(H)) > 0
                spd_flag = 1;
            end
            c_spd(j,1) = c_spd(j,1)+1;
        end
    end
    
end

% print matrix properties

for i = 1:3
    for j=1:nn
        
        if i == 1
            if j == nn
                fprintf(fid1,'%d\\\\\n',j);
            else
                fprintf(fid1,'%d &',j);
            end
        elseif i == 2
            if j == nn
                fprintf(fid1,'%d \\\\\n',rc(j,1));
            else
                fprintf(fid1,'%d &',rc(j,1));
            end
        else
            if j == nn
                fprintf(fid1,'%d \\\\\n',rc(j,3));
            else
                fprintf(fid1,'%d &',rc(j,3));
            end
        end
    end
end
%
% for j=1:nn
%     mi = indlist(j);
%     fprintf(fid1,'%d & %d & %d \\\\\n',j...
%                     ,rc(j,1),rc(j,3));
% end
fprintf(fid1,'\n'); fprintf(fid1,'\n');


fclose(fid1);
