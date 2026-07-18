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
Y = y;
%X = data;

% BCD_MVRL3 expects X{v} as d_v × n (features × samples)
for i = 1:length(X)
    X{i} = zscore(X{i}');       % → d_v × n
end

c = length(unique(Y));
n = length(Y);
V = length(X);

fprintf('Dataset: %s | Samples: %d | Classes: %d | Views: %d\n', ...
        dataName, n, c, V);
for i = 1:V
    fprintf('  View %d: %d×%d (d×n)\n', i, size(X{i}, 1), size(X{i}, 2));
end

% 确保添加评估指标函数所在的文件夹路径
addpath(genpath(fullfile(scriptDir, '..', 'measure')));

%% ================== Grid Search ==================
fprintf('\n\n=== Grid Search ===\n');
m_grid      = [c, 2*c, 3*c,7*c];
lambda_grid = [0.001 0.01, 0.1, 1, 10, 100];
gamma_grid  = [0.001, 0.01, 0.1, 1, 10,100];

% 取消了数据维度安全过滤机制，强制使用用户定义的 m_grid
fprintf('m range: [%s]\n', num2str(m_grid));
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        'm', 'lambda', 'gamma', 'ACC', 'NMI', 'Obj', 'Time');
fprintf('%-6s %-8s %-8s %-10s %-10s %-10s %-8s\n', ...
        '--', '------', '-----', '---', '---', '---', '----');

opts_q = struct('max_iter', 30, 'tol', 1e-4, 'verbose', false);
best_acc = 0; best_nmi = 0;
best_m = 0; best_lambda = 0; best_gamma = 0;
total = length(m_grid)*length(lambda_grid)*length(gamma_grid);
cnt = 0;

for mi = 1:length(m_grid)
    for li = 1:length(lambda_grid)
        for gi = 1:length(gamma_grid)
            cnt = cnt + 1;
            try
                t0 = tic;
                % 网格搜索阶段调用核心算法
                [~, ~, ~, ~, Sq, obj_q] = bcd_mvrl4(X, m_grid(mi), c, ...
                    lambda_grid(li), gamma_grid(gi), opts_q);
                t1 = toc(t0);
                
                % 评估聚类结果
                res_q = myNMIACCwithmean(Sq', Y, c);
                
                fprintf('%-6d %-8.2f %-8.3f %-10.4f %-10.4f %-10.2e %6.1fs [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), gamma_grid(gi), ...
                        res_q(1), res_q(2), obj_q(end), t1, cnt, total);
                        
                % 记录最佳参数
                if res_q(1) > best_acc
                    best_acc = res_q(1); best_nmi = res_q(2);
                    best_m = m_grid(mi); best_lambda = lambda_grid(li); best_gamma = gamma_grid(gi);
                end
            catch ME
                fprintf('%-6d %-8.2f %-8.3f FAILED: %s [%d/%d]\n', ...
                        m_grid(mi), lambda_grid(li), gamma_grid(gi), ME.message, cnt, total);
            end
        end
    end
end

% 容错处理：如果全部报错，给定一组默认兜底参数
if best_m == 0
    best_m = max(c, 8); best_lambda = 1; best_gamma = 0.1;
    fprintf('\n>>> All failed. Using defaults: m=%d, lambda=%.2f, gamma=%.3f\n', ...
            best_m, best_lambda, best_gamma);
end

fprintf('\nBest Parameters found: m=%d, lambda=%.2f, gamma=%.3f | ACC=%.4f, NMI=%.4f\n', ...
        best_m, best_lambda, best_gamma, best_acc, best_nmi);

%% ================== Refined Run with Best Params ==================
fprintf('\n=== Refined Run (Computing Optimal Result) ===\n');
opts.max_iter = 100; opts.verbose = true;

tic;
% 利用网格搜索出的最优超参数运行最终的精调
[R_b, Ps_b, Q_b, A_b, S_b, obj_b] = bcd_mvrl4(X, best_m, c, best_lambda, best_gamma, opts);
t_b = toc;

res_b = myNMIACCwithmean(S_b', Y, c);
fprintf('\nRefined Result: ACC=%.4f, NMI=%.4f, Purity=%.4f, F=%.4f, Time=%.1fs\n', ...
        res_b(1), res_b(2), res_b(3), res_b(4), t_b);

% 绘制收敛曲线
figure;
semilogy(obj_b, 'r-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective');
title(sprintf('BCD-MVRL4 Best (m=%d,\\lambda=%.2f,\\gamma=%.3f) ACC=%.4f', ...
      best_m, best_lambda, best_gamma, best_acc));
grid on;