% demo_bcd8.m  —  BCD-MVRL10 演示 (非负Q + NMF乘法更新A)
%
%  特性:
%    - Q ≥ 0: 非负视图偏移
%    - A 更新: NMF 乘法更新 + 单纯形投影
%    - Ps 更新: 闭式解 (R^T R ≈ I 近似)
%    - Q 更新: 闭式解 + max(·,0) 非负投影
%    - β||Q A||_F^2: 对 QA 乘积的 Frobenius 正则

clear; clc; warning off;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(genpath(fullfile(scriptDir, '..')));
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% ================== 加载数据集 ==================
dataName = 'NGs_fea';
fprintf('加载数据集: %s\n', dataName);
dsPath = 'D:\BaiduNetdiskDownload\Multi-view datasets\';
load([dsPath dataName]);
Y = gt;
X=data;

for i = 1:length(X)
    X{i} = zscore(X{i});                          % z-score 标准化
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
m_grid      = [c, 2*c, 3*c];
lambda_grid = [0.01, 0.1, 1, 10, 100];      % 共识
beta_grid   = [1 10 100 1000 10000];    % QA 正则

fprintf('m: [%s]  λ: [%s]  β: [%s]\n', ...
        num2str(m_grid), num2str(lambda_grid), num2str(beta_grid));
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-5s %-8s\n', ...
        'm', 'λ', 'β', 'ACC', 'NMI', 'Obj', '收敛', 'Time');
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-5s %-8s\n', ...
        '--', '----', '----', '---', '---', '---', '----', '----');

opts_q = struct('max_iter', 30, 'tol', 1e-4, 'verbose', false);
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
                [~, ~, ~, ~, Sq, obj_q, diag_q] = bcd_mvrl10(X, m_grid(mi), c, ...
                    lambda_grid(li), beta_grid(bi), opts_q);
                t1 = toc(t0);
                res_q = myNMIACCwithmean(Sq', Y, c);
                conv_str = iif(diag_q.converged, '✓', '✗');

                fprintf('%-6d %-8.2f %-8.3f %-10.4f %-10.4f %-10.2e %-5s %6.1fs [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ...
                        res_q(1), res_q(2), obj_q(end), conv_str, t1, cnt, total);

                if res_q(1) > best_acc
                    best_acc = res_q(1); best_nmi = res_q(2);
                    best_m = m_grid(mi); best_lambda = lambda_grid(li); best_beta = beta_grid(bi);
                end
            catch ME
                fprintf('%-6d %-8.2f %-8.3f 失败: %s [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), beta_grid(bi), ME.message, cnt, total);
            end
        end
    end
end

if best_m == 0
    best_m = max(c, 8); best_lambda = 1; best_beta = 0.1;
end

fprintf('\n最优: m=%d, λ=%.2f, β=%.3f | ACC=%.4f, NMI=%.4f\n', ...
        best_m, best_lambda, best_beta, best_acc, best_nmi);

%% ================== 精调运行 ==================
fprintf('\n=== 精调运行 (每轮目标值) ===\n');
opts.max_iter = 100; opts.verbose = false;  % 关闭自带输出，手动打印

tic;
[R_b, Ps_b, Q_b, A_b, S_b, obj_b, diag_b] = bcd_mvrl10(X, best_m, c, best_lambda, best_beta, opts);
t_b = toc;

% 打印每轮目标函数值
fprintf('  迭代   目标函数值       相对变化\n');
fprintf('  ----  --------------  ------\n');
for i = 1:length(obj_b)
    if i == 1
        rc = 0;
    else
        rc = abs(obj_b(i-1)-obj_b(i)) / max(abs(obj_b(i-1)), 1e-8);
    end
    marker = '';
    if i == diag_b.final_iter, marker = ' ← 收敛'; end
    fprintf('  %4d  %14.6e  %6.2e%s\n', i, obj_b(i), rc, marker);
end

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\n精调结果: ACC=%.4f NMI=%.4f Purity=%.4f F=%.4f 时间=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);
fprintf('收敛状态: %s (iter=%d, rel=%.2e)\n', ...
        iif(diag_b.converged,'已收敛','未收敛'), diag_b.final_iter, diag_b.final_rel);
fprintf('各项占比: 重建=%.1f%%  共识=%.1f%%  QA=%.1f%%\n', ...
        100*diag_b.obj_terms(1)/obj_b(end), ...
        100*diag_b.obj_terms(2)/obj_b(end), ...
        100*diag_b.obj_terms(3)/obj_b(end));

figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('目标函数值');
title(sprintf('BCD-MVRL10 (m=%d,\\lambda=%.2f,\\beta=%.3f) ACC=%.4f', ...
      best_m, best_lambda, best_beta, best_acc));
grid on;

%% ==================== 辅助 ====================
function s = iif(cond, s_true, s_false)
if cond, s = s_true; else, s = s_false; end
end
