% demo_mvc.m  —  Demo script for Multi-View Clustering via Alternating Optimization
%
%  This script demonstrates the MVC-AO algorithm on the Handwritten dataset.
%  The algorithm jointly learns:
%    R  (n×c)   — shared soft cluster assignment (row-stochastic)
%    Ps (c×m)   — shared cluster→anchor prototype (orthogonal)
%    Q  (c×m)   — view-specific prototype offsets
%    A  (m×d_v) — view-specific anchor dictionaries
%
%  Reconstruction:  X^{(v)} ≈ R × (Ps + Q^{(v)}) × A^{(v)}

clear; clc; warning off;

% Add paths relative to script location
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));

%% ================== Load Dataset ==================
dataName = 'CCV_fea';
fprintf('Loading dataset: %s\n', dataName);
dsPath = 'D:\BaiduNetdiskDownload\Multi-view datasets\';
load([dsPath dataName]);

% Data preparation
% Raw .mat: X{i} is already n × d_i (samples × features) — NO transpose needed
%
% Y = y;

% Normalize each view to [0,1] range
for i = 1:length(X)
    X{i} = normalize(X{i}, 'range');   % normalize columns of n × d_i
end

c = length(unique(Y));     % number of clusters
n = length(Y);

fprintf('Dataset: %s | Samples: %d | Classes: %d | Views: %d\n', ...
        dataName, n, c, length(X));
for i = 1:length(X)
    fprintf('  View %d: %d×%d (n × d)\n', i, size(X{i}, 1), size(X{i}, 2));
end

%% ================== Run MVC-AO ==================
% Hyperparameters
m_val     = 3 * c;         % anchors = 3k (same scale as RCAGL)
alpha_val = 0.1;           % Q  regularization
gamma_val = 0.1;           % A  regularization

fprintf('\n>>> Running MVC-AO: m=%d, alpha=%.2f, gamma=%.2f\n', m_val, alpha_val, gamma_val);

opts = struct();
opts.max_iter    = 100;
opts.pgd_steps   = 5;
opts.cg_max_iter = 30;
opts.cg_tol      = 1e-5;
opts.tol         = 1e-4;
opts.verbose     = true;

tic;
[R, Ps, Q, A_cell, obj_hist] = mvc_ao(X, c, m_val, alpha_val, gamma_val, opts);
elapsed = toc;

fprintf('Time: %.2f seconds\n', elapsed);

%% ================== Evaluate Clustering ==================
% Cluster assignment from R (argmax of each row)
[~, y_pred] = max(R, [], 2);

% Evaluate using existing metrics
addpath(genpath('../measure/'));
res = myNMIACCwithmean(R, Y, c);

fprintf('\n=== Clustering Results ===\n');
fprintf('ACC : %.4f\n', res(1));
fprintf('NMI : %.4f\n', res(2));
fprintf('Purity : %.4f\n', res(3));
fprintf('F-score: %.4f\n', res(4));

%% ================== Objective Convergence Plot ==================
figure;
semilogy(obj_hist, 'b-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective Value');
title(sprintf('MVC-AO Convergence (m=%d, \\alpha=%.2f, \\gamma=%.2f)', m_val, alpha_val, gamma_val));
grid on;

%% ================== Optional: Hyperparameter Grid Search ==================
fprintf('\n\n=== Quick Grid Search ===\n');
m_grid     = [c, 2*c, 3*c];
alpha_grid = [0.01, 0.1, 1];
gamma_grid = [0.01, 0.1, 1];

fprintf('%-8s %-8s %-8s %-10s %-10s %-10s\n', 'm', 'alpha', 'gamma', 'ACC', 'NMI', 'Obj');
fprintf('%-8s %-8s %-8s %-10s %-10s %-10s\n', '--', '-----', '-----', '---', '---', '---');

best_acc = 0; best_nmi = 0;

for mi = 1:length(m_grid)
    for ai = 1:length(alpha_grid)
        for gi = 1:length(gamma_grid)
            opts_q = opts;
            opts_q.verbose = false;

            try
                [Rq, ~, ~, ~, obj_q] = mvc_ao(X, c, m_grid(mi), alpha_grid(ai), gamma_grid(gi), opts_q);
                [~, yq] = max(Rq, [], 2);
                res_q = myNMIACCwithmean(Rq, Y, c);
                fprintf('%-8d %-8.2f %-8.2f %-10.4f %-10.4f %-10.2e\n', ...
                        m_grid(mi), alpha_grid(ai), gamma_grid(gi), res_q(1), res_q(2), obj_q(end));

                if res_q(1) > best_acc
                    best_acc = res_q(1); best_nmi = res_q(2);
                end
            catch ME
                fprintf('%-8d %-8.2f %-8.2f FAILED: %s\n', ...
                        m_grid(mi), alpha_grid(ai), gamma_grid(gi), ME.message);
            end
        end
    end
end

fprintf('\nBest: ACC=%.4f, NMI=%.4f\n', best_acc, best_nmi);
