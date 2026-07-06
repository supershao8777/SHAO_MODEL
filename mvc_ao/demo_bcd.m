% demo_bcd.m  —  Demo for BCD Multi-View Representation Learning
%
%  Jointly learns:
%    R  (n×r)   — shared orthogonal representation  [R^T R = I_r]
%    Ps (r×m)   — shared orthogonal prototype       [Ps^T Ps = I_m]
%    Q  (r×m)   — view-specific non-negative offset  [Q ≥ 0]
%    A  (m×d_v) — non-negative anchor dictionary     [A ≥ 0]
%
%  Reconstruction: X^{(v)} ≈ R × (Ps + Q^{(v)}) × A^{(v)}
%
%  L1 redundancy penalty: α Σ_{v<u} ||Q^{(v)} ⊙ Q^{(u)}||₁

clear; clc; warning off;

% Add paths
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));

%% ================== Load Dataset ==================
dataName = 'NUS-WIDE-OBJ';
fprintf('Loading dataset: %s\n', dataName);
dsPath = 'D:\BaiduNetdiskDownload\Multi-view datasets\';
load([dsPath dataName]);
%X=data;
% Data preparation: X{v} is n × d_v
Y = y;

% Z-score normalize each view
for i = 1:length(X)
    X{i} = zscore(X{i});
end

c = length(unique(Y));     % number of clusters
n = length(Y);
V = length(X);

fprintf('Dataset: %s | Samples: %d | Classes: %d | Views: %d\n', ...
        dataName, n, c, V);
for i = 1:V
    fprintf('  View %d: %d×%d (n×d)\n', i, size(X{i}, 1), size(X{i}, 2));
end

%% ================== Run BCD-MVRL ==================
% Latent dimension r (must be ≥ m for orthogonality Ps^T Ps = I_m)
r_val     = max(c, 32);       % latent dimension
m_val     = c;                % prototypes (= number of classes)
alpha_val = 0.1;              % L1 redundancy penalty
gamma_val = 0.1;              % ridge on A

fprintf('\n>>> Running BCD-MVRL: r=%d, m=%d, alpha=%.2f, gamma=%.2f\n', ...
        r_val, m_val, alpha_val, gamma_val);

opts = struct();
opts.max_iter  = 100;
opts.pgd_steps = 3;           % inner PGD iterations for Ps and Q
opts.tol       = 1e-4;
opts.verbose   = true;

tic;
[R, Ps, Q, A_cell, obj_hist] = bcd_mvrl(X, r_val, m_val, alpha_val, gamma_val, opts);
elapsed = toc;
fprintf('Time: %.2f seconds\n', elapsed);

%% ================== Evaluate Clustering ==================
% Use R as spectral embedding, run k-means
addpath(genpath(fullfile(scriptDir, '..', 'measure')));
res = myNMIACCwithmean(R, Y, c);

fprintf('\n=== Clustering Results (via k-means on R) ===\n');
fprintf('ACC    : %.4f\n', res(1));
fprintf('NMI    : %.4f\n', res(2));
fprintf('Purity : %.4f\n', res(3));
fprintf('F-score: %.4f\n', res(4));

%% ================== Objective Convergence Plot ==================
figure;
semilogy(obj_hist, 'b-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective Value');
title(sprintf('BCD-MVRL Convergence (r=%d, m=%d, \\alpha=%.2f, \\gamma=%.2f)', ...
      r_val, m_val, alpha_val, gamma_val));
grid on;

%% ================== Quick Grid Search ==================
fprintf('\n\n=== Quick Grid Search ===\n');
r_grid     = [max(c,16), max(c,32), max(c,64)];
m_grid     = [c, 2*c];
alpha_grid = [0.001 0.01, 0.1, 1 10];
gamma_grid = [0.001 0.01, 0.1, 1 10];

fprintf('%-6s %-5s %-8s %-8s %-10s %-10s %-10s\n', ...
        'r', 'm', 'alpha', 'gamma', 'ACC', 'NMI', 'Obj');
fprintf('%-6s %-5s %-8s %-8s %-10s %-10s %-10s\n', ...
        '--', '---', '-----', '-----', '---', '---', '---');

best_acc = 0; best_nmi = 0;

for ri = 1:length(r_grid)
    for mi = 1:length(m_grid)
        if m_grid(mi) > r_grid(ri), continue; end   % need m ≤ r
        for ai = 1:length(alpha_grid)
            for gi = 1:length(gamma_grid)
                opts_q = opts;
                opts_q.verbose = false;

                try
                    [Rq, ~, ~, ~, obj_q] = bcd_mvrl(X, r_grid(ri), m_grid(mi), ...
                        alpha_grid(ai), gamma_grid(gi), opts_q);
                    res_q = myNMIACCwithmean(Rq, Y, c);
                    fprintf('%-6d %-5d %-8.2f %-8.2f %-10.4f %-10.4f %-10.2e\n', ...
                            r_grid(ri), m_grid(mi), alpha_grid(ai), gamma_grid(gi), ...
                            res_q(1), res_q(2), obj_q(end));

                    if res_q(1) > best_acc
                        best_acc = res_q(1); best_nmi = res_q(2);
                    end
                catch ME
                    fprintf('%-6d %-5d %-8.2f %-8.2f FAILED: %s\n', ...
                            r_grid(ri), m_grid(mi), alpha_grid(ai), gamma_grid(gi), ME.message);
                end
            end
        end
    end
end

fprintf('\nBest: ACC=%.4f, NMI=%.4f\n', best_acc, best_nmi);
