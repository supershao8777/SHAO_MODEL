% demo_bcd2.m  —  Demo for BCD Multi-View Representation Learning v2
%
%  Jointly learns:
%    R{v} (d_v×m) — view-specific orthogonal projection   [R^T R = I_m]
%    Ps   (m×c)   — shared orthogonal prototype           [Ps^T Ps = I_c]
%    Q{v} (m×c)   — view-specific offset in null(Ps^T)    [Q^T Ps = 0]
%    A{v} (c×n)   — column-stochastic anchor coefficients  [≥0, col-sum=1]
%    S    (m×n)   — consensus representation              [≥0, col-sum=1]
%
%  Data format: X^{(v)} is d_v × n (features × samples)
%
%  Objective:
%    Σ_v ||X^{(v)}-R^{(v)}(Ps+Q^{(v)})A^{(v)}||² + λ||(Ps+Q^{(v)})A^{(v)}-S||² + γ||A^{(v)}||²

clear; clc; warning off;

% Add paths
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));

%% ================== Load Dataset ==================
dataName = 'ALOI';
fprintf('Loading dataset: %s\n', dataName);
dsPath = 'D:\BaiduNetdiskDownload\Multi-view datasets\';
load([dsPath dataName]);
%X = data;
Y = y;

% NOTE: BCD_MVRL2 expects X{v} as d_v × n (features × samples)
% Transpose if data is n × d_v
for i = 1:length(X)
    X{i} = zscore(X{i}');       % → d_v × n, z-score normalize
end

c = length(unique(Y));           % number of clusters
n = length(Y);
V = length(X);

fprintf('Dataset: %s | Samples: %d | Classes: %d | Views: %d\n', ...
        dataName, n, c, V);
for i = 1:V
    fprintf('  View %d: %d×%d (d×n)\n', i, size(X{i}, 1), size(X{i}, 2));
end

%% ================== Run BCD-MVRL2 ==================


opts = struct();
opts.max_iter  = 50;
opts.pgd_steps = 3;             % inner PGD iterations for Ps and Q
opts.tol       = 1e-4;
opts.verbose   = true;


%% ================== Hyperparameter Grid Search ==================
fprintf('\n\n=== Hyperparameter Grid Search ===\n');

% Parameter grids
m_grid      = [c 2*c 3*c];           % projection dim
lambda_grid = [0.01, 0.1, 1, 10, 100];                     % consensus weight
gamma_grid  = [0.001, 0.01, 0.1, 1, 10];                  % ridge on A

% Validate m ≤ min(d_v) for each view (soft — solver handles d_v<m gracefully)
min_dv = inf;
for i = 1:V
    min_dv = min(min_dv, size(X{i}, 1));
end
% Filter to feasible range, but always keep at least one value
m_grid = m_grid(m_grid <= min_dv);
if isempty(m_grid)
    m_grid = min_dv;   % fallback: use the smallest feature dimension
end

fprintf('m search range (≤ min(d_v)=%d): [%s]\n', min_dv, num2str(m_grid));

fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        'm', 'lambda', 'gamma', 'ACC', 'NMI', 'Obj', 'Time');
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '--', '------', '-----', '---', '---', '---', '----');

best_acc = 0;  best_nmi = 0;
best_params = [];
best_m = 0; best_lambda = 0; best_gamma = 0;

opts_q = opts;
opts_q.verbose = false;
opts_q.max_iter = 30;               % fewer iters for speed in grid search

total_runs = length(m_grid) * length(lambda_grid) * length(gamma_grid);
run_count = 0;

for mi = 1:length(m_grid)
    for li = 1:length(lambda_grid)
        for gi = 1:length(gamma_grid)
            run_count = run_count + 1;

            try
                t_start = tic;
                [Rq, ~, ~, ~, Sq, obj_q] = bcd_mvrl2(X, m_grid(mi), c, ...
                    lambda_grid(li), gamma_grid(gi), opts_q);
                t_elapsed = toc(t_start);

                res_q = myNMIACCwithmean(Sq', Y, c);

                fprintf('%-6d %-8.2f %-8.3f %-10.4f %-10.4f %-10.2e %6.1fs [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), gamma_grid(gi), ...
                        res_q(1), res_q(2), obj_q(end), t_elapsed, ...
                        run_count, total_runs);

                if res_q(1) > best_acc
                    best_acc = res_q(1);  best_nmi = res_q(2);
                    best_m      = m_grid(mi);
                    best_lambda = lambda_grid(li);
                    best_gamma  = gamma_grid(gi);
                end
            catch ME
                fprintf('%-6d %-8.2f %-8.3f FAILED: %s [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), gamma_grid(gi), ...
                        ME.message, run_count, total_runs);
            end
        end
    end
end

% Fallback: if all runs failed, use feasible default
if best_m == 0
    fprintf('\n>>> All grid runs failed. Using feasible default. <<<\n');
    best_m      = min(max(c, 8), min_dv);
    best_lambda = 1;
    best_gamma  = 0.1;
end

fprintf('\n========================================\n');
fprintf('Best Result:\n');
fprintf('  m=%d, lambda=%.2f, gamma=%.3f\n', ...
        best_m, best_lambda, best_gamma);
fprintf('  ACC=%.4f, NMI=%.4f\n', best_acc, best_nmi);

%% ================== Refined Run with Best Params ==================
fprintf('\n\n=== Refined Run with Best Parameters ===\n');
opts.max_iter = 100;
opts.pgd_steps = 5;
opts.verbose = true;

tic;
[R_best, Ps_best, Q_best, A_best, S_best, obj_best] = ...
    bcd_mvrl2(X, best_m, c, best_lambda, best_gamma, opts);
elapsed_best = toc;

res_best = myNMIACCwithmean(S_best', Y, c);
fprintf('\nRefined Results (max_iter=%d, pgd_steps=%d):\n', opts.max_iter, opts.pgd_steps);
fprintf('  ACC=%.4f, NMI=%.4f, Purity=%.4f, F-score=%.4f, Time=%.1fs\n', ...
        res_best(1), res_best(2), res_best(3), res_best(4), elapsed_best);

figure;
semilogy(obj_best, 'r-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective Value');
title(sprintf('Best BCD-MVRL2 (m=%d, \\lambda=%.2f, \\gamma=%.3f)  ACC=%.4f', ...
      best_m, best_lambda, best_gamma, best_acc));
grid on;
