% demo_bcd6.m  —  Demo for BCD-MVRL8 (IRLS + L21 on Q*A)
%
%  Objective:
%    min Σ_v ||X^{(v)}-R^{(v)}(Ps+Q^{(v)})A^{(v)}||² + λ||(Ps+Q^{(v)})A^{(v)}-S||² + β||Q^{(v)}A^{(v)}||_{2,1}
%
%  Constraints: R*R^T=I, Ps^T Ps=I, A/S col-stochastic
%  Regularizer: L21 on Q*A (IRLS with reweighted least squares)
%
%  NOTE: Need d_v ≤ m for R*R^T=I_{d_v}. Choose m ≥ max(d_v).

clear; clc; warning off;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% ================== Load Dataset ==================
dataName = 'Wikipedia';
fprintf('Loading dataset: %s\n', dataName);
dsPath = 'D:\BaiduNetdiskDownload\Multi-view datasets\';
load([dsPath dataName]);
Y = y;
%X = data;

for i = 1:length(X)
    X{i} = zscore(X{i}');          % → d_v × n
end

c = length(unique(Y));
n = length(Y);
V = length(X);

% Find min d_v for m lower bound (need m ≥ max(d_v) ideally)
min_dv = inf; max_dv = 0;
for i = 1:V
    dv = size(X{i},1);
    min_dv = min(min_dv, dv); max_dv = max(max_dv, dv);
end

fprintf('Dataset: %s | Samples: %d | Classes: %d | Views: %d\n', ...
        dataName, n, c, V);
fprintf('d_v range: [%d, %d]\n', min_dv, max_dv);
for i = 1:V
    fprintf('  View %d: %d×%d\n', i, size(X{i},1), size(X{i},2));
end

%% ================== Grid Search ==================
fprintf('\n\n=== Grid Search ===\n');
% m must be ≥ d_v for R*R^T=I. Use max_dv as the base.
%m_base = max(max_dv, c);
m_grid      = [c,2*c,3*c];
lambda_grid = [0.01, 0.1, 1, 10, 100];
beta_grid   = [0.001, 0.01, 0.1, 1, 10];

fprintf('m: [%s]  lambda: [%s]  beta: [%s]\n', ...
        num2str(m_grid), num2str(lambda_grid), num2str(beta_grid));
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        'm', 'lambda', 'beta', 'ACC', 'NMI', 'Obj', 'Time');
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '--', '------', '----', '---', '---', '---', '----');

opts_q = struct('max_iter', 30, 'pgd_steps', 3, 'irls_iters', 3, ...
                'tol', 1e-4, 'verbose', false);
best_acc = 0; best_nmi = 0;
best_m = 0; best_lambda = 0; best_beta = 0;
total = length(m_grid) * length(lambda_grid) * length(beta_grid);
cnt = 0;

for mi = 1:length(m_grid)
    for li = 1:length(lambda_grid)
        for bi = 1:length(beta_grid)
            cnt = cnt + 1;
            try
                t0 = tic;
                [~, ~, ~, ~, Sq, obj_q] = bcd_mvrl8(X, m_grid(mi), c, ...
                    lambda_grid(li), beta_grid(bi), opts_q);
                t1 = toc(t0);
                res_q = myNMIACCwithmean(Sq', Y, c);

                fprintf('%-6d %-8.2f %-8.3f %-10.4f %-10.4f %-10.2e %6.1fs [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ...
                        res_q(1), res_q(2), obj_q(end), t1, cnt, total);

                if res_q(1) > best_acc
                    best_acc = res_q(1); best_nmi = res_q(2);
                    best_m = m_grid(mi); best_lambda = lambda_grid(li); best_beta = beta_grid(bi);
                end
            catch ME
                fprintf('%-6d %-8.2f %-8.3f FAILED: %s [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ME.message, cnt, total);
            end
        end
    end
end

if best_m == 0
    best_m = m_base; best_lambda = 1; best_beta = 0.1;
end

fprintf('\nBest: m=%d, lambda=%.2f, beta=%.3f | ACC=%.4f, NMI=%.4f\n', ...
        best_m, best_lambda, best_beta, best_acc, best_nmi);

%% ================== Refined Run ==================
fprintf('\n=== Refined Run with Best Params ===\n');
opts.max_iter = 100; opts.pgd_steps = 5; opts.irls_iters = 5; opts.verbose = true;

tic;
[R_b, Ps_b, Q_b, A_b, S_b, obj_b] = bcd_mvrl8(X, best_m, c, best_lambda, best_beta, opts);
t_b = toc;

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\nRefined: ACC=%.4f NMI=%.4f Purity=%.4f F=%.4f Time=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);

figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective');
title(sprintf('BCD-MVRL8 Best (m=%d,\\lambda=%.2f,\\beta=%.3f) ACC=%.4f', ...
      best_m, best_lambda, best_beta, best_acc));
grid on;
