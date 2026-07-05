clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'Handwritten'};
dsPath = '.\datasets\';


for dsi =1:1:length(ds)
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    
    X=X';
    Y=y;
    % for i=1:length(X)
    %     X{i}=X{i}';
    % end
    % 
    
    k = length(unique(Y));
    n = length(Y);
    
     lambda1 = [0.01 0.1 1 10 100 ];
     rho=[ 0.01 0.1 1 10]
    % lambda1 =0;
      lambda2 = [ 0.01 0.1 1 10 100];
      
     anchor = [k,2*k,3*k];
%     lambda = [1  ];
%     anchor = [k ];
        
    %%
    allresult = []; 
      save_name=['modify_v4_' ds{dsi}   '.mat'];
   for id3 = 1:length(rho)
    for ichor = 1:length(anchor)
        for id = 1:length(lambda1)
            for id2 = 1:length(lambda2)
                tic;
                [A, P, C, D, iter, obj, y1] = rcagl(X,Y,anchor(ichor),lambda1(id),lambda2(id2),rho(id3));

                [UU,~,V]=svd(C','econ');
                res = myNMIACCwithmean(UU,Y,k);
                t = toc;
            fprintf('Rho:%.2f Anchor:%d \t Lambda1:%d\t Lambda2:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[rho(id3) anchor(ichor) lambda1(id) lambda2(id2) res(1) res(2) res(3) res(4) t]);
            allresult = [allresult; rho(id3) anchor(ichor) res t];
            end
        end
    end
   end
   save(save_name,'allresult')
end


