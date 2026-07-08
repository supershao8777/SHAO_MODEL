function [R, Ps, Q, A, S, r, obj_history] = bcd_mvrl9(X, m, c, lambda1, lambda2, lambda3, opts)
% BCD_MVRL9  多视图表示学习（视图权重 + PGD-A + Sylvester-Q + R正则）
%
%   [R, Ps, Q, A, S, r, obj] = bcd_mvrl9(X, m, c, lambda1, lambda2, lambda3, opts)
%
%   目标函数：
%     L = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ₁ Σ_v r_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2        ← 加权共识
%       + λ₂ Σ_v ||Ps^T Q^{(v)}||_F^2                       ← Ps-Q 正交惩罚
%       + λ₃ Σ_v ||R^{(v)}||_F^2                            ← R 正则化
%
%   约束：
%     Ps  : m×c,   Ps^T Ps = I_c                  (Stiefel 流形)
%     A{v}: c×n,   A{v} ≥ 0, 每列求和=1           (列概率单纯形)
%     S   : m×n,   S ≥ 0, 每列求和=1              (列概率单纯形)
%     r_v : 标量,  r_v ≥ 0, Σ r_v = 1             (视图权重)
%     R{v}, Q{v}: 无约束
%
%   更新顺序：A → Ps → Q → R → S → r
%
%   数据格式：X{v} 为 d_v × n  (特征 × 样本)

%% 参数解析
if nargin < 7, opts = struct(); end
max_iter   = get_opt(opts, 'max_iter',   50);
pgd_steps  = get_opt(opts, 'pgd_steps',  5);   % A/Ps/Q 内迭代步数
tol        = get_opt(opts, 'tol',        1e-4);
verbose    = get_opt(opts, 'verbose',    true);

%% 维度
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

if verbose
    fprintf('\n=== BCD-MVRL9 初始化 ===\n');
    fprintf('样本: %d,  投影维数 m: %d,  锚点数 c: %d,  视图数 V: %d\n', n, m, c, V);
    fprintf('λ₁=%.4f (加权共识), λ₂=%.4f (Ps-Q正交惩罚), λ₃=%.4f (R正则)\n', ...
            lambda1, lambda2, lambda3);
    fprintf('约束: Ps^T Ps=I, A/S 列单纯形, r_v≥0 且 Σr_v=1\n');
    fprintf('更新顺序: A → Ps → Q → R → S → r\n');
end

for v = 1:V
    assert(size(X{v},2) == n, '所有视图的样本数必须相同。');
end

%% 初始化
rng(42, 'twister');

% --- R^{(v)} (d_v × m, 随机小值) ---
R = cell(1, V);
for v = 1:V
    R{v} = randn(size(X{v},1), m) * 0.01;
end

% --- Ps (m × c, 标准正交列) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);           % Ps^T Ps = I_c
else
    Ps = randn(m, c) / sqrt(m);
end

% --- Q^{(v)} (m × c, 小随机值) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = randn(m, c) * 0.01;
end

% --- A^{(v)} (c × n, 列单纯形) ---
A = cell(1, V);
for v = 1:V
    A{v} = col_simplex_project(rand(c, n));
end

% --- S (m × n, 列单纯形) ---
S = col_simplex_project(rand(m, n));

% --- r_v (视图权重, 初始等权) ---
r = ones(V, 1) / V;

if verbose
    fprintf('初始化完成。\n');
    fprintf('\n  迭代   目标函数值       相对变化   重建误差    共识项      PsQ项       RegR       [视图权重]\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------  ---------  -------------\n');
end

%% ==================== 主优化循环 ====================
obj_history = zeros(max_iter, 1);

for iter = 1:max_iter
    %% Step 1: 更新 A^{(v)}  (投影梯度下降 + 单纯形投影)
    %   ∇A = 2 M^T R^T (R M A - X) + 2 λ₁ r_v M^T (M A - S)
    %   Lipschitz 步长
    for v = 1:V
        Mv = Ps + Q{v};                          % m × c
        RtR = R{v}' * R{v};                      % m × m

        L_A = 2 * norm(Mv'*RtR*Mv, 2) + 2*lambda1*r(v)*norm(Mv'*Mv, 2);
        eta_A = 1 / max(L_A, epsilon);

        for pgd = 1:pgd_steps
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};        % d_v × n
            E2 = Mv * A{v} - S;                   % m × n
            grad_A = 2 * Mv' * R{v}' * E1 + 2*lambda1*r(v) * Mv' * E2;  % c × n

            A{v} = A{v} - eta_A * grad_A;         % 梯度步
            A{v} = col_simplex_project(A{v});     % 列单纯形投影
        end
    end

    %% Step 2: 更新 Ps  (梯度下降 + SVD Stiefel 投影)
    %   ∇Ps = 2 Σ_v [R_v^T(R_v M_v A_v - X_v)A_v^T]
    %        + 2λ₁ Σ_v [r_v (M_v A_v - S) A_v^T]
    %        + 2λ₂ Σ_v [Q_v Q_v^T Ps]
    S_AA = zeros(c, c);  H_QQ = zeros(m, m);  max_W = 0;
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';
        H_QQ = H_QQ + Q{v} * Q{v}';
        max_W = max(max_W, norm(R{v}'*R{v}, 2));
    end
    L_Ps = 2*max_W*norm(S_AA,2) + 2*lambda1*norm(S_AA,2) + 2*lambda2*norm(H_QQ,2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';           % 重建
            E2 = Mv * A{v} - S;
            grad_Ps = grad_Ps + lambda1 * r(v) * E2 * A{v}';  % 加权共识
        end
        grad_Ps = 2 * grad_Ps;
        % Ps-Q 正交惩罚梯度
        for v = 1:V
            grad_Ps = grad_Ps + 2*lambda2 * Q{v} * (Q{v}' * Ps);
        end

        Ps = Ps - eta_Ps * grad_Ps;
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';                                    % Stiefel 投影
    end

    %% Step 3: 更新 Q^{(v)}  (Kronecker Sylvester 精确解)
    %   E·Q·F + G·Q = C
    %   E = R_v^T R_v + λ₁ r_v I_m,  F = A_v A_v^T,  G = λ₂ Ps Ps^T
    %   C = R_v^T X_v A_v^T - E·Ps·F + λ₁ r_v S A_v^T
    for v = 1:V
        E_Q = R{v}' * R{v} + lambda1 * r(v) * eye(m);   % m × m
        F_Q = A{v} * A{v}';                               % c × c
        G_Q = lambda2 * (Ps * Ps');                       % m × m

        % 右端项 C
        C_Q = R{v}' * X{v} * A{v}' + lambda1*r(v)*S*A{v}' - E_Q*Ps*F_Q;  % m × c

        % Kronecker: (F^T ⊗ E + I_c ⊗ G) vec(Q) = vec(C)
        K = kron(F_Q', E_Q) + kron(eye(c), G_Q);          % mc × mc
        q_vec = K \ C_Q(:);                                % mc × 1
        Q{v} = reshape(q_vec, [m, c]);                    % m × c
    end

    %% Step 4: 更新 R^{(v)}  (岭回归，含 λ₃ 正则)
    %   R_v = X_v Y_v^T (Y_v Y_v^T + λ₃ I)^{-1}
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                          % m × n
        R{v} = X{v} * Yv' / (Yv * Yv' + lambda3 * eye(m)); % d_v × m
    end

    %% Step 5: 更新 S  (加权平均 + 列单纯形投影)
    S_mean = zeros(m, n);
    for v = 1:V
        Mv = Ps + Q{v};                                   % m × c
        S_mean = S_mean + r(v) * Mv * A{v};               % 加权平均
    end
    S = col_simplex_project(S_mean);

    %% Step 6: 更新 r_v  (逆误差加权)
    %   E_v = ||M_v A_v - S||_F^2
    %   r_v = (1/(E_v+ε)) / Σ_u (1/(E_u+ε))
    E_r = zeros(V, 1);
    for v = 1:V
        Mv = Ps + Q{v};
        E_r(v) = norm(Mv * A{v} - S, 'fro')^2;
    end
    inv_E = 1 ./ (E_r + epsilon);
    r = inv_E / sum(inv_E);                               % 归一化

    %% 计算目标函数
    obj_rec  = 0;  obj_cons = 0;  obj_psq = 0;  obj_regR = 0;
    for v = 1:V
        Mv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
        obj_cons = obj_cons + r(v) * norm(Mv * A{v} - S, 'fro')^2;
        obj_psq  = obj_psq  + norm(Ps' * Q{v}, 'fro')^2;
        obj_regR = obj_regR + norm(R{v}, 'fro')^2;
    end
    obj = obj_rec + lambda1 * obj_cons + lambda2 * obj_psq + lambda3 * obj_regR;
    obj_history(iter) = obj;

    %% 收敛判断
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose
        if mod(iter,5)==1 || iter==1
            fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e  %9.2e  [r:', ...
                    iter, obj, rel_change, obj_rec, lambda1*obj_cons, ...
                    lambda2*obj_psq, lambda3*obj_regR);
            fprintf(' %.2f', r);
            fprintf(']\n');
        end
    end

    if iter > 1 && rel_change < tol
        if verbose
            fprintf('  --- 迭代 %d 收敛 (相对变化 < %.0e) ---\n', iter, tol);
            fprintf('  最终视图权重 r_v: ');
            fprintf('%.3f ', r);
            fprintf('\n');
        end
        obj_history = obj_history(1:iter);
        break;
    end
end

if verbose && iter >= max_iter
    fprintf('  达到最大迭代次数 (%d)。\n', max_iter);
end

%% 汇总
if verbose
    fprintf('\n========== 汇总 ==========\n');
    fprintf('迭代: %d,  目标: %.4e\n', iter, obj);
    fprintf('重建:    %14.6e  (%5.1f%%)\n', obj_rec, 100*obj_rec/obj);
    fprintf('共识:    %14.6e  (%5.1f%%)\n', lambda1*obj_cons, 100*lambda1*obj_cons/obj);
    fprintf('PsQ:     %14.6e  (%5.1f%%)\n', lambda2*obj_psq, 100*lambda2*obj_psq/obj);
    fprintf('RegR:    %14.6e  (%5.1f%%)\n', lambda3*obj_regR, 100*lambda3*obj_regR/obj);
    fprintf('视图权重 r_v: ');
    fprintf('%.3f ', r);
    fprintf('\n');
    fprintf('||Ps^T Ps-I|| = %.2e\n', norm(Ps'*Ps - eye(c), 'fro'));
    % Q 统计
    for v = 1:V
        fprintf('  Q{%d}: norm=%.3e, nnz_rows=%d\n', v, norm(Q{v},'fro'), ...
                sum(sqrt(sum(Q{v}.^2,2)) > epsilon));
    end
    % R 统计
    for v = 1:V
        fprintf('  R{%d}: norm=%.3e\n', v, norm(R{v},'fro'));
    end
end
end

%% ==================== 辅助函数 ====================

function S_out = col_simplex_project(S_in)
% 列单纯形投影：每列 ≥0 且求和为 1
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
