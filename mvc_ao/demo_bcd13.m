% demo_bcd13.m  —  BCD-MVRL16 (S Orthogonal + Q≥0 + Ps Stiefel)
%
%  Objective: Σ||X-R(Ps+Q)A||²+β||PsA-S||²+γ||Ps^T Q||²
%  S: S S^T=I_m, Q≥0, Ps: Stiefel tangent+SVD

clear; clc; warning off;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);addpath(genpath(fullfile(scriptDir,'..')));
addpath(genpath(fullfile(scriptDir,'..','measure')));

dataName='Caltech101-20';fprintf('Loading: %s\n',dataName);
load(['D:\BaiduNetdiskDownload\Multi-view datasets\' dataName]);
Y=y;
for i=1:length(X),X{i}=zscore(X{i}');end
c=length(unique(Y));n=length(Y);V=length(X);
fprintf('Samples:%d Classes:%d Views:%d\n',n,c,V);

%% Grid (3D: m, β, γ)
fprintf('\n=== Grid ===\n');
m_grid=[c,2*c,3*c];beta_grid=[0.01,0.1,1,10];gam_grid=[0.001,0.01,0.1,1];
fprintf('%-4s %-7s %-7s %-9s %-9s %-9s %-7s\n','m','β','γ','ACC','NMI','Obj','Time');
fprintf('%-4s %-7s %-7s %-9s %-9s %-9s %-7s\n','--','----','----','---','---','---','----');
opts_q=struct('max_iter',30,'tol',1e-4,'verbose',false);
best_acc=0;best_nmi=0;best=struct('m',0,'b',0,'g',0);
cnt=0;total=length(m_grid)*length(beta_grid)*length(gam_grid);
for mi=1:length(m_grid)
    for bi=1:length(beta_grid)
        for gi=1:length(gam_grid)
            cnt=cnt+1;try
                t0=tic;[~,~,~,~,Sq,obj_q]=bcd_mvrl16(X,m_grid(mi),c,beta_grid(bi),gam_grid(gi),opts_q);
                t1=toc(t0);res_q=myNMIACCwithmean(Sq',Y,c);
                fprintf('%-4d %-7.2f %-7.3f %-9.4f %-9.4f %-9.2e %5.1fs [%d/%d]\n',...
                        m_grid(mi),beta_grid(bi),gam_grid(gi),res_q(1),res_q(2),obj_q(end),t1,cnt,total);
                if res_q(1)>best_acc,best_acc=res_q(1);best_nmi=res_q(2);
                    best.m=m_grid(mi);best.b=beta_grid(bi);best.g=gam_grid(gi);end
            catch ME,fprintf('%-4d %-7.2f %-7.3f FAILED:%s [%d/%d]\n',...
                        m_grid(mi),beta_grid(bi),gam_grid(gi),ME.message,cnt,total);end
        end
    end
end
if best.m==0,best.m=2*c;best.b=1;best.g=0.1;end
fprintf('\nBest: m=%d β=%.2f γ=%.3f | ACC=%.4f NMI=%.4f\n',best.m,best.b,best.g,best_acc,best_nmi);

%% Refined
fprintf('\n=== Refined ===\n');opts.max_iter=100;opts.verbose=true;
tic;[~,~,~,~,S_b,obj_b]=bcd_mvrl16(X,best.m,c,best.b,best.g,opts);t_b=toc;
res_b=myNMIACCwithmean(S_b',Y,c);
fprintf('\nACC=%.4f NMI=%.4f Purity=%.4f F=%.4f Time=%.1fs\n',res_b(1),res_b(2),res_b(3),res_b(4),t_b);
figure;semilogy(obj_b,'r-','LineWidth',1.5);xlabel('Iter');ylabel('Obj');
title(sprintf('BCD-MVRL16 m=%d β=%.2f γ=%.3f ACC=%.4f',best.m,best.b,best.g,best_acc));grid on;
