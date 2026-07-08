% demo_bcd9.m  —  BCD-MVRL11 演示 (Riemannian Ps + closed-Q + Duchi simplex)
%
%  特性:
%    - Ps: Riemannian梯度 + QR收缩 (Stiefel流形)
%    - Q:  闭式解 + max(·,0) 非负投影
%    - A:  单步PGD + Duchi单纯形投影 (非简单归一化)
%    - S:  均值 + Duchi单纯形投影
%    - R:  无约束伪逆

clear; clc; warning off;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% ================== 加载数据集 ==================
dataName = 'Wikipedia';
fprintf('加载数据集: %s\n', dataName);
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

fprintf('数据集: %s | 样本: %d | 类别: %d | 视图: %d\n', dataName, n, c, V);
for i = 1:V
    fprintf('  视图 %d: %d×%d\n', i, size(X{i},1), size(X{i},2));
end

%% ================== 网格搜索 ==================
fprintf('\n\n=== 网格搜索 ===\n');
m_grid      = [c 2*c 3*c ];
lambda_grid = [0.01, 0.1, 1, 10, 100];
beta_grid   = [0.01 0.1 1, 10, 100, 1000];

fprintf('m: [%s]  λ: [%s]  β: [%s]\n', ...
        num2str(m_grid), num2str(lambda_grid), num2str(beta_grid));
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        'm', 'λ', 'β', 'ACC', 'NMI', 'Obj', 'Time');
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '--', '----', '----', '---', '---', '---', '----');

opts_q = struct('max_iter', 30, 'eta_ps', 1e-3, 'verbose', false);
best_acc = 0; best_nmi = 0;
best_m = 0; best_lambda = 0; best_beta = 0;
total = length(m_grid)*length(lambda_grid)*length(beta_grid);
cnt = 0;

for mi = 1:length(m_grid)
    for li = 1:length(lambda_grid)
        for bi = 1:length(beta_grid)
            cnt = cnt + 1;
            try
                t0 = tic;
                [~, ~, ~, ~, Sq, obj_q] = bcd_mvrl11(X, m_grid(mi), c, ...
                    lambda_grid(li), beta_grid(bi), opts_q);
                t1 = toc(t0);
                res_q = myNMIACCwithmean(Sq', Y, c);

                fprintf('%-6d %-8.2f %-8.1f %-10.4f %-10.4f %-10.2e %6.1fs [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ...
                        res_q(1), res_q(2), obj_q(end), t1, cnt, total);

                if res_q(1) > best_acc
                    best_acc = res_q(1); best_nmi = res_q(2);
                    best_m = m_grid(mi); best_lambda = lambda_grid(li); best_beta = beta_grid(bi);
                end
            catch ME
                fprintf('%-6d %-8.2f %-8.1f 失败: %s [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ME.message, cnt, total);
            end
        end
    end
end

if best_m == 0
    best_m = max(c, 8); best_lambda = 1; best_beta = 10;
end

fprintf('\n最优: m=%d, λ=%.2f, β=%.1f | ACC=%.4f, NMI=%.4f\n', ...
        best_m, best_lambda, best_beta, best_acc, best_nmi);

%% ================== 精调运行 ==================
fprintf('\n=== 精调运行 ===\n');
opts.max_iter = 200; opts.verbose = true;

tic;
[R_b, Ps_b, Q_b, A_b, S_b, obj_b] = bcd_mvrl11(X, best_m, c, best_lambda, best_beta, opts);
t_b = toc;

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\n精调结果: ACC=%.4f NMI=%.4f Purity=%.4f F=%.4f 时间=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);

figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('目标函数值');
title(sprintf('BCD-MVRL11 (m=%d,\\lambda=%.2f,\\beta=%.1f) ACC=%.4f', ...
      best_m, best_lambda, best_beta, best_acc));
grid on;
