function [R, Ps, Q, A, S, obj_history] = bcd_mvrl4(X, m, c, lambda, gamma, opts)
% BCD_MVRL4  Multi-View Representation Learning (orthogonal R).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl4(X, m, c, lambda, gamma, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%         + γ Σ_v ||A^{(v)}||_F^2
%
%   Constraints:
%     R{v}: d_v×m,   (R{v})^T R{v} = I_m                   (orthogonal columns)
%     Ps  : m×c,     Ps^T Ps = I_c                          (orthogonal prototype)
%     Q{v}: m×c,     unconstrained                          (view-specific offset)
%     A{v}: c×n,     A{v} ≥ 0, (A{v})^T 1_c = 1_n         (column-stochastic)
%     S   : m×n,     S ≥ 0, S^T 1_m = 1_n                  (col-stochastic consensus)
%
%   With R^T R = I_m, the Ps and Q gradients simplify to closed-form
%   bilateral division (no gradient descent needed).
%
%   Data format: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter = get_opt(opts, 'max_iter', 50);
tol      = get_opt(opts, 'tol',      1e-4);
verbose  = get_opt(opts, 'verbose',  true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

if verbose
    fprintf('\n=== BCD-MVRL4 Initialization ===\n');
    fprintf('Samples: %d, Proj: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, gamma: %.4f\n', lambda, gamma);
end

for v = 1:V
    [dv, nv] = size(X{v});
    assert(nv == n, 'All views must have same sample count.');
    if dv < m && verbose
        fprintf('  Note: View %d (d_v=%d < m=%d), R{%d} orthogonality relaxed.\n', v, dv, m, v);
    end
end
if m < c && verbose
    fprintf('  Note: m=%d < c=%d, Ps orthogonality relaxed.\n', m, c);
end

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, orthonormal columns if d_v ≥ m) ---
R = cell(1, V);
for v = 1:V
    dv = size(X{v}, 1);
    if dv >= m
        [R{v}, ~] = qr(randn(dv, m), 0);          % d_v × m, R^T R = I_m
    else
        R{v} = randn(dv, m) / sqrt(dv);            % approx unit-norm cols
    end
end

% --- Ps (m × c, orthonormal columns if m ≥ c) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);                  % m × c, Ps^T Ps = I_c
else
    Ps = randn(m, c) / sqrt(m);
end

% --- Q^{(v)} (m × c, small random) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = randn(m, c) * 0.01;
end

% --- S (m × n, column-stochastic) ---
S = col_simplex_project(rand(m, n));

% --- A^{(v)} (c × n, column-stochastic) ---
A = cell(1, V);
for v = 1:V
    A{v} = col_simplex_project(rand(c, n));
end

if verbose
    fprintf('Init done.  Update order: S → A → R → Ps → Q\n');
    fprintf('\n  Iter   Objective       RelChg    Recon        Consensus\n');
    fprintf('  ----  --------------  ------   ----------   ----------\n');
end

%% ==================== BCD Main Loop ====================
obj_history = zeros(max_iter, 1);

% Pre-compute once per iteration to avoid repeated work
for iter = 1:max_iter
    %% Step 1: Update S (consensus, column-simplex)
    %   S = simplex( (1/V) Σ_v (Ps+Q_v) A_v )
    S_mean = zeros(m, n);
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        S_mean = S_mean + Mv * A{v};                 % m × n
    end
    S = col_simplex_project(S_mean / V);

    %% Step 2: Update A^{(v)} (closed-form + simplex, uses R^T R = I)
    %   M = Ps+Q_v  [m×c]
    %   A = ((1+λ) M^T M + γ I)^{-1} (M^T R^T X + λ M^T S)
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c

        % R^T R = I simplifies the reconstruction Hessian
        M_A = (1 + lambda) * (Mv' * Mv) + gamma * eye(c);     % c × c
        rhs_A = Mv' * R{v}' * X{v} + lambda * Mv' * S;         % c × n
        A{v} = M_A \ rhs_A;                                     % c × n

        A{v} = col_simplex_project(A{v});            % ≥0, col-sum=1
    end

    %% Step 3: Update R^{(v)} (Orthogonal Procrustes, enforces R^T R = I_m)
    %   Y_v = (Ps+Q_v) A_v  [m × n]
    %   [U,~,V] = svd(X_v Y_v^T) → R_v = U V^T
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                     % m × n
        [U_R, ~, V_R] = svd(X{v} * Yv', 'econ');     % d_v × m
        R{v} = U_R * V_R';                           % d_v × m, columns orthonormal
    end

    %% Step 4: Update Ps (closed-form + SVD, uses R^T R = I)
    %   (1+λ) Ps (Σ_v A_v A_v^T) = G_Ps
    %   G_Ps = Σ_v [R_v^T X_v A_v^T + λ S A_v^T] - (1+λ) Σ_v Q_v A_v A_v^T
    %   Ps_unc = G_Ps / ((1+λ) S_AA + εI)
    %   [U,~,V] = svd(Ps_unc) → Ps = U V^T
    S_AA = zeros(c, c);
    G_Ps = zeros(m, c);
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';                  % c × c

        G_Ps = G_Ps + R{v}' * X{v} * A{v}';          % m × c  (recon)
        G_Ps = G_Ps + lambda * S * A{v}';             % m × c  (consensus)
        G_Ps = G_Ps - (1 + lambda) * Q{v} * (A{v} * A{v}');  % offset
    end

    Ps_unc = G_Ps / ((1 + lambda) * S_AA + epsilon * eye(c));  % m × c

    % SVD projection → Stiefel manifold
    [U_ps, ~, V_ps] = svd(Ps_unc, 'econ');
    Ps = U_ps * V_ps';                               % m × c, Ps^T Ps = I_c

    %% Step 5: Update Q^{(v)} (closed-form, uses R^T R = I)
    %   (1+λ) Q_v (A_v A_v^T) = G_Qv
    %   G_Qv = R_v^T X_v A_v^T + λ S A_v^T - (1+λ) Ps A_v A_v^T
    for v = 1:V
        S_Av = A{v} * A{v}';                         % c × c

        G_Qv = R{v}' * X{v} * A{v}';                 % m × c  (recon)
        G_Qv = G_Qv + lambda * S * A{v}';            % m × c  (consensus)
        G_Qv = G_Qv - (1 + lambda) * Ps * S_Av;      % m × c  (Ps term)

        Q{v} = G_Qv / ((1 + lambda) * S_Av + epsilon * eye(c));  % m × c
    end

    %% Objective
    obj_rec  = 0;  obj_cons = 0;  obj_regA = 0;
    for v = 1:V
        Mv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
        obj_cons = obj_cons + norm(Mv * A{v} - S, 'fro')^2;
        obj_regA = obj_regA + norm(A{v}, 'fro')^2;
    end
    obj = obj_rec + lambda * obj_cons + gamma * obj_regA;
    obj_history(iter) = obj;

    %% Convergence
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %10.2e\n', ...
                iter, obj, rel_change, obj_rec, lambda*obj_cons);
    end

    if iter > 1 && rel_change < tol
        if verbose, fprintf('  Converged at iter %d.\n', iter); end
        obj_history = obj_history(1:iter);
        break;
    end
end

if verbose && iter >= max_iter
    fprintf('  Max iters (%d).\n', max_iter);
end

%% Summary
if verbose
    fprintf('\n=== Summary ===\n');
    fprintf('Iter: %d, Obj: %.4e | Recon: %.4e  Cons: %.4e  Ridge: %.4e\n', ...
            iter, obj, obj_rec, lambda*obj_cons, gamma*obj_regA);

    err_R = 0;
    for v = 1:V
        err_R = err_R + norm(R{v}'*R{v} - eye(m), 'fro')^2;
    end
    fprintf('||R^T R-I||_avg = %.2e,  ||Ps^T Ps-I|| = %.2e\n', ...
            sqrt(err_R/V), norm(Ps'*Ps - eye(c), 'fro'));
end
end

%% ==================== Helpers ====================

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
