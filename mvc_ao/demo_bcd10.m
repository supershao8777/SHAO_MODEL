% demo_bcd10.m  —  BCD-MVRL19 (doubly-stochastic C + Sinkhorn)
%
%  Objective:
%    Σ||X-R(Ps+Q)Z||²+λ||C Ps Z-S||²+μ⟨D,C⟩+γ||Z||²

clear; clc; warning off;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% Load dataset
dataName = 'Animal';
fprintf('Loading: %s\n', dataName);
load(['D:\BaiduNetdiskDownload\Multi-view datasets\' dataName]);
Y = y; %X = data;
for i = 1:length(X), X{i} = zscore(X{i}'); end

c = length(unique(Y)); n = length(Y); V = length(X);
fprintf('Samples: %d, Classes: %d, Views: %d\n', n, c, V);

%% Grid search
fprintf('\n=== Grid Search ===\n');
m_grid   = [c, 2*c, 3*c];
lam_grid = [0.001,0.01, 0.1, 1, 10, 100,1000];

fprintf('%-4s %-7s %-9s %-9s %-9s %-7s\n', 'm','λ','ACC','NMI','Obj','Time');
fprintf('%-4s %-7s %-9s %-9s %-9s %-7s\n', '--','----','---','---','---','----');

opts_q = struct('max_iter', 30, 'pgd_steps', 3, 'tol', 1e-4, 'verbose', false);
best_acc = 0; best_nmi = 0; best_m = 0; best_l = 0;
cnt = 0; total = length(m_grid)*length(lam_grid);

for mi = 1:length(m_grid)
    for li = 1:length(lam_grid)
        cnt = cnt + 1;
        try
            t0 = tic;
            [~, ~, ~, ~, Sq, ~, obj_q] = bcd_mvrl19(X, m_grid(mi), c, lam_grid(li), 0.1, 0.01, opts_q);
            t1 = toc(t0);
            res_q = myNMIACCwithmean(Sq', Y, c);

            fprintf('%-4d %-7.2f %-9.4f %-9.4f %-9.2e %5.1fs [%d/%d]\n', ...
                    m_grid(mi), lam_grid(li), res_q(1), res_q(2), obj_q(end), t1, cnt, total);

            if res_q(1) > best_acc
                best_acc = res_q(1); best_nmi = res_q(2);
                best_m = m_grid(mi); best_l = lam_grid(li);
            end
        catch ME
            fprintf('%-4d %-7.2f FAILED: %s [%d/%d]\n', ...
                    m_grid(mi), lam_grid(li), ME.message, cnt, total);
        end
    end
end

if best_m == 0, best_m = 2*c; best_l = 1; end
fprintf('\nBest: m=%d, λ=%.2f | ACC=%.4f, NMI=%.4f\n', best_m, best_l, best_acc, best_nmi);

%% Refined run
fprintf('\n=== Refined Run ===\n');
opts.max_iter = 200; opts.pgd_steps = 5; opts.alpha_P = 1e-3; opts.alpha_Z = 1e-3;
opts.verbose = true;

tic;
[R_b, Ps_b, Q_b, Z_b, S_b, ~, obj_b] = bcd_mvrl19(X, best_m, c, best_l, 0.1, 0.01, opts);
t_b = toc;

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\nRefined: ACC=%.4f NMI=%.4f Purity=%.4f F=%.4f Time=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);

figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('Iter'); ylabel('Loss');
title(sprintf('BCD-MVRL19 (m=%d,\\lambda=%.2f) ACC=%.4f', best_m, best_l, best_acc));
grid on;
