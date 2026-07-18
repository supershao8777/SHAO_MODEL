% demo_bcd15.m  —  BCD-MVRL18 (Laplacian from Ps + full consensus)
%
%  Objective: Σ||X-R(Ps+Q)A||²+λ||(Ps+Q)A-S||²+γ||A||²+βTr(SLS^T)
%  L built from Ps rows, Q≥0, S: simplex+Laplacian

clear; clc; warning off;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);addpath(genpath(fullfile(scriptDir,'..')));
addpath(genpath(fullfile(scriptDir,'..','measure')));

dataName='Handwritten';fprintf('Loading: %s\n',dataName);
load(['D:\BaiduNetdiskDownload\Multi-view datasets\' dataName]);
Y=y;
%X=data;
for i=1:length(X),X{i}=zscore(X{i}');end
c=length(unique(Y));n=length(Y);V=length(X);
fprintf('Samples:%d Classes:%d Views:%d\n',n,c,V);

%% Grid
fprintf('\n=== Grid ===\n');
m_grid=[4*c];lam_grid=[0.01 1 10,100];gam_grid=[0.01,0.1,1,10];beta_grid=[0.01,0.1,1,10,100,1000];
fprintf('%-4s %-6s %-6s %-6s %-9s %-9s %-9s %-7s\n','m','λ','γ','β','ACC','NMI','Obj','Time');
fprintf('%-4s %-6s %-6s %-6s %-9s %-9s %-9s %-7s\n','--','----','----','----','---','---','---','----');
opts_q=struct('max_iter',30,'tol',1e-4,'verbose',false);
best_acc=0;best_nmi=0;best=struct('m',0,'l',0,'g',0,'b',0);
cnt=0;total=length(m_grid)*length(lam_grid)*length(gam_grid)*length(beta_grid);
for mi=1:length(m_grid)
    for li=1:length(lam_grid)
        for gi=1:length(gam_grid)
            for bi=1:length(beta_grid)
                cnt=cnt+1;try
                    t0=tic;[~,~,~,~,Sq,obj_q]=bcd_mvrl18(X,m_grid(mi),c,lam_grid(li),gam_grid(gi),beta_grid(bi),opts_q);
                    t1=toc(t0);res_q=myNMIACCwithmean(Sq',Y,c);
                    fprintf('%-4d %-6.2f %-6.2f %-6.2f %-9.4f %-9.4f %-9.2e %5.1fs [%d/%d]\n',...
                            m_grid(mi),lam_grid(li),gam_grid(gi),beta_grid(bi),res_q(1),res_q(2),obj_q(end),t1,cnt,total);
                    if res_q(1)>best_acc,best_acc=res_q(1);best_nmi=res_q(2);
                        best.m=m_grid(mi);best.l=lam_grid(li);best.g=gam_grid(gi);best.b=beta_grid(bi);end
                catch ME,fprintf('%-4d %-6.2f %-6.2f %-6.2f FAILED:%s [%d/%d]\n',...
                            m_grid(mi),lam_grid(li),gam_grid(gi),beta_grid(bi),ME.message,cnt,total);end
            end
        end
    end
end
if best.m==0,best.m=2*c;best.l=1;best.g=0.1;best.b=0.1;end
fprintf('\nBest: m=%d λ=%.2f γ=%.2f β=%.2f | ACC=%.4f NMI=%.4f\n',best.m,best.l,best.g,best.b,best_acc,best_nmi);

%% Refined
fprintf('\n=== Refined ===\n');opts.max_iter=100;opts.verbose=true;
tic;[~,~,~,~,S_b,obj_b]=bcd_mvrl18(X,best.m,c,best.l,best.g,best.b,opts);t_b=toc;
res_b=myNMIACCwithmean(S_b',Y,c);
fprintf('\nACC=%.4f NMI=%.4f Purity=%.4f F=%.4f Time=%.1fs\n',res_b(1),res_b(2),res_b(3),res_b(4),t_b);
figure;semilogy(obj_b,'r-','LineWidth',1.5);xlabel('Iter');ylabel('Obj');
title(sprintf('BCD-MVRL18 m=%d λ=%.2f γ=%.2f β=%.2f ACC=%.4f',best.m,best.l,best.g,best.b,best_acc));grid on;
