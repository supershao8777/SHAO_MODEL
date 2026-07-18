% demo_bcd12.m  —  BCD-MVRL15 (HQ + L1 Hadamard Ps⊙Q)
%
%  Objective: Σ √(||X-R(Ps+Q)A||²+ε) + λ||Ps⊙Q||₁ + γ||A||²
%  Q: weighted soft-threshold, w_v: HQ adaptive weights

clear; clc; warning off;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);addpath(genpath(fullfile(scriptDir,'..')));
addpath(genpath(fullfile(scriptDir,'..','measure')));

dataName='Caltech101-7';fprintf('Loading: %s\n',dataName);
load(['D:\BaiduNetdiskDownload\Multi-view datasets\' dataName]);
Y=y;%X=data;
for i=1:length(X),X{i}=zscore(X{i}');end
c=length(unique(Y));n=length(Y);V=length(X);
fprintf('Samples:%d Classes:%d Views:%d\n',n,c,V);

%% Grid
fprintf('\n=== Grid ===\n');
m_grid=[c,2*c,3*c];lam_grid=[0.01,0.1,1,10];gam_grid=[0.001,0.01,0.1,1];
fprintf('%-4s %-7s %-7s %-9s %-9s %-9s %-7s\n','m','λ','γ','ACC','NMI','Obj','Time');
fprintf('%-4s %-7s %-7s %-9s %-9s %-9s %-7s\n','--','----','----','---','---','---','----');
opts_q=struct('max_iter',30,'tol',1e-5,'verbose',false);
best_acc=0;best_nmi=0;best=struct('m',0,'l',0,'g',0);
cnt=0;total=length(m_grid)*length(lam_grid)*length(gam_grid);
for mi=1:length(m_grid)
    for li=1:length(lam_grid)
        for gi=1:length(gam_grid)
            cnt=cnt+1;try
                t0=tic;[~,~,~,A_q,~,obj_q]=bcd_mvrl15(X,m_grid(mi),c,lam_grid(li),gam_grid(gi),opts_q);
                t1=toc(t0);Am=0;for vv=1:V,Am=Am+A_q{vv}';end;res_q=myNMIACCwithmean(Am/V,Y,c);
                fprintf('%-4d %-7.2f %-7.3f %-9.4f %-9.4f %-9.2e %5.1fs [%d/%d]\n',...
                        m_grid(mi),lam_grid(li),gam_grid(gi),res_q(1),res_q(2),obj_q(end),t1,cnt,total);
                if res_q(1)>best_acc,best_acc=res_q(1);best_nmi=res_q(2);
                    best.m=m_grid(mi);best.l=lam_grid(li);best.g=gam_grid(gi);end
            catch ME,fprintf('%-4d %-7.2f %-7.3f FAILED:%s [%d/%d]\n',...
                        m_grid(mi),lam_grid(li),gam_grid(gi),ME.message,cnt,total);end
        end
    end
end
if best.m==0,best.m=2*c;best.l=1;best.g=0.1;end
fprintf('\nBest: m=%d λ=%.2f γ=%.3f | ACC=%.4f NMI=%.4f\n',best.m,best.l,best.g,best_acc,best_nmi);

%% Refined
fprintf('\n=== Refined ===\n');opts.max_iter=100;opts.verbose=true;
tic;[~,~,~,A_b,w_b,obj_b]=bcd_mvrl15(X,best.m,c,best.l,best.g,opts);t_b=toc;
Amb=0;for vv=1:V,Amb=Amb+A_b{vv}';end;res_b=myNMIACCwithmean(Amb/V,Y,c);
fprintf('\nACC=%.4f NMI=%.4f Purity=%.4f F=%.4f Time=%.1fs\n',res_b(1),res_b(2),res_b(3),res_b(4),t_b);
fprintf('HQ weights w_v:%s\n',mat2str(w_b',3));
figure;semilogy(obj_b,'r-','LineWidth',1.5);xlabel('Iter');ylabel('Obj');
title(sprintf('BCD-MVRL15 m=%d λ=%.2f γ=%.3f ACC=%.4f',best.m,best.l,best.g,best_acc));grid on;
