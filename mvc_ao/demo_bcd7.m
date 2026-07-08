% demo_bcd7.m  —  BCD-MVRL9 演示 (视图权重 + PGD-A + Sylvester-Q)
%
%  新增特性:
%    - r_v: 视图权重 (逆误差自适应)
%    - A 更新: 投影梯度下降 + 单纯形投影
%    - Q 更新: Kronecker Sylvester 精确解
%    - 加权共识: λ₁ Σ r_v ||M_v A_v - S||²

clear; clc; warning off;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% ================== 加载数据集 ==================
dataName = 'Caltech101-7';
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
m_grid       = [c, 2*c, 3*c];
lambda1_grid = [0.01, 0.1, 1, 10, 100];     % 共识权重
lambda2_grid = [ 0.01, 0.1, 1, 10];   % Ps-Q 正交惩罚
lambda3_grid = [ 0.01, 0.1,1,10];           % R 正则化

fprintf('m: [%s]  λ₁: [%s]  λ₂: [%s]  λ₃: [%s]\n', ...
        num2str(m_grid), num2str(lambda1_grid), num2str(lambda2_grid), num2str(lambda3_grid));
fprintf('%-6s %-8s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        'm', 'λ₁', 'λ₂', 'λ₃', 'ACC', 'NMI', 'Obj', 'Time');
fprintf('%-6s %-8s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '--', '----', '----', '----', '---', '---', '---', '----');

opts_q = struct('max_iter', 30, 'pgd_steps', 3, 'tol', 1e-4, 'verbose', false);
best_acc = 0; best_nmi = 0;
best_m = 0; best_l1 = 0; best_l2 = 0; best_l3 = 0;
total = length(m_grid)*length(lambda1_grid)*length(lambda2_grid)*length(lambda3_grid);
cnt = 0;

for mi = 1:length(m_grid)
    for l1i = 1:length(lambda1_grid)
        for l2i = 1:length(lambda2_grid)
            for l3i = 1:length(lambda3_grid)
                cnt = cnt + 1;
                try
                    t0 = tic;
                    [~, ~, ~, ~, Sq, ~, obj_q] = bcd_mvrl9(X, m_grid(mi), c, ...
                        lambda1_grid(l1i), lambda2_grid(l2i), lambda3_grid(l3i), opts_q);
                    t1 = toc(t0);
                    res_q = myNMIACCwithmean(Sq', Y, c);

                    fprintf('%-6d %-8.2f %-8.3f %-8.3f %-10.4f %-10.4f %-10.2e %6.1fs [%d/%d]\n', ...
                            m_grid(mi), lambda1_grid(l1i), lambda2_grid(l2i), lambda3_grid(l3i), ...
                            res_q(1), res_q(2), obj_q(end), t1, cnt, total);

                    if res_q(1) > best_acc
                        best_acc = res_q(1); best_nmi = res_q(2);
                        best_m = m_grid(mi); best_l1 = lambda1_grid(l1i);
                        best_l2 = lambda2_grid(l2i); best_l3 = lambda3_grid(l3i);
                    end
                catch ME
                    fprintf('%-6d %-8.2f %-8.3f %-8.3f 失败: %s [%d/%d]\n', ...
                            m_grid(mi), lambda1_grid(l1i), lambda2_grid(l2i), lambda3_grid(l3i), ...
                            ME.message, cnt, total);
                end
            end
        end
    end
end

if best_m == 0
    best_m = max(c, 8); best_l1 = 1; best_l2 = 0.1; best_l3 = 0.01;
end

fprintf('\n最优: m=%d, λ₁=%.2f, λ₂=%.3f, λ₃=%.3f | ACC=%.4f, NMI=%.4f\n', ...
        best_m, best_l1, best_l2, best_l3, best_acc, best_nmi);

%% ================== 精调运行 ==================
fprintf('\n=== 精调运行 (最优参数) ===\n');
opts.max_iter = 100; opts.pgd_steps = 5; opts.verbose = true;

tic;
[R_b, Ps_b, Q_b, A_b, S_b, r_b, obj_b] = bcd_mvrl9(X, best_m, c, best_l1, best_l2, best_l3, opts);
t_b = toc;

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\n精调结果: ACC=%.4f NMI=%.4f Purity=%.4f F=%.4f 时间=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);
fprintf('最终视图权重: ');
fprintf('%.3f ', r_b);
fprintf('\n');

figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('目标函数值');
title(sprintf('BCD-MVRL9 (m=%d,\\lambda_1=%.2f,\\lambda_2=%.3f,\\lambda_3=%.3f) ACC=%.4f', ...
      best_m, best_l1, best_l2, best_l3, best_acc));
grid on;
