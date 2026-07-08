function [R, Ps, Q, A, S, obj_history] = bcd_mvrl6(X, m, c, lambda, beta, opts)
% BCD_MVRL6  Multi-View Representation Learning (Stiefel Ps + Sylvester Q).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl6(X, m, c, lambda, beta, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%         + β Σ_v ||Ps^T Q^{(v)}||_F^2
%
%   Constraints:
%     S   : m×n,  S ≥ 0, S^T 1_m = 1_n                   (col-stochastic)
%     A{v}: c×n,  A{v} ≥ 0, (A{v})^T 1_c = 1_n          (col-stochastic)
%     Ps  : m×c,  Ps^T Ps = I_c                           (Stiefel manifold)
%     R{v}, Q{v}: UNCONSTRAINED
%
%   Updates:
%     S  — simplex-projected mean
%     A  — closed-form ridge + simplex
%     R  — pseudo-inverse
%     Ps — gradient descent + SVD Stiefel projection
%     Q  — Kronecker-vectorized Sylvester solver  (exact closed-form)
%
%   Data format: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

if verbose
    fprintf('\n=== BCD-MVRL6 Initialization ===\n');
    fprintf('Samples: %d, Proj: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, beta: %.4f\n', lambda, beta);
    fprintf('Constraints: Ps^T Ps=I_c (Stiefel), A/S col-stochastic\n');
end

for v = 1:V
    assert(size(X{v},2) == n, 'All views must have same sample count.');
end
if m < c && verbose
    fprintf('  Note: m=%d < c=%d, Ps orthogonality relaxed.\n', m, c);
end

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, random small) ---
R = cell(1, V);
for v = 1:V
    R{v} = randn(size(X{v},1), m) * 0.01;
end

% --- Ps (m × c, orthonormal columns if m ≥ c) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);                   % Ps^T Ps = I_c
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
    fprintf('Init done.  Order: S → A → R → Ps → Q\n');
    fprintf('\n  Iter   Objective       RelChg    Recon        Cons       PsQ\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------\n');
end

%% ==================== BCD Main Loop ====================
obj_history = zeros(max_iter, 1);

for iter = 1:max_iter
    %% Step 1: Update S (column-simplex mean)
    S_mean = zeros(m, n);
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        S_mean = S_mean + Mv * A{v};                 % m × n
    end
    S = col_simplex_project(S_mean / V);

    %% Step 2: Update A^{(v)} (closed-form + simplex)
    %   A = (M^T R^T R M + λ M^T M + εI)^{-1} (M^T R^T X + λ M^T S)
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        RtR = R{v}' * R{v};                          % m × m

        M_A = Mv' * RtR * Mv + lambda * (Mv' * Mv) + epsilon * eye(c);  % c × c
        rhs_A = Mv' * R{v}' * X{v} + lambda * Mv' * S;                   % c × n
        A{v} = M_A \ rhs_A;                                               % c × n
        A{v} = col_simplex_project(A{v});            % ≥0, col-sum=1
    end

    %% Step 3: Update R^{(v)} (pseudo-inverse)
    %   Y_v = (Ps+Q_v) A_v  [m × n]
    %   R_v = X_v Y_v^T (Y_v Y_v^T)^{-1}
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                     % m × n
        R{v} = X{v} * Yv' / (Yv * Yv' + epsilon * eye(m));  % d_v × m
    end

    %% Step 4: Update Ps (gradient descent + SVD Stiefel projection)
    %   ∇Ps = 2 Σ_v [R_v^T(R_v M_v A_v - X_v)A_v^T]          (recon)
    %        + 2λ Σ_v [(M_v A_v - S)A_v^T]                    (consensus)
    %        + 2β Σ_v [Q_v Q_v^T Ps]                          (PsQ penalty)
    %
    %   Lipschitz: L_Ps = 2 max(||W_v||)·||Σ Sv|| + 2λ·||Σ Sv|| + 2β·||Σ Q_v Q_v^T||

    S_AA  = zeros(c, c);     % Σ A_v A_v^T
    H_QQ  = zeros(m, m);     % Σ Q_v Q_v^T
    max_W = 0;
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';
        H_QQ = H_QQ + Q{v} * Q{v}';
        max_W = max(max_W, norm(R{v}' * R{v}, 2));
    end
    L_Ps = 2 * max_W * norm(S_AA,2) + 2*lambda * norm(S_AA,2) + 2*beta * norm(H_QQ,2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};            % d_v × n
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';   % m × c  (recon)
            E2 = Mv * A{v} - S;                       % m × n
            grad_Ps = grad_Ps + lambda * E2 * A{v}';  % m × c  (consensus)
        end
        grad_Ps = 2 * grad_Ps;

        % PsQ penalty: 2β Q_v Q_v^T Ps
        for v = 1:V
            grad_Ps = grad_Ps + 2 * beta * Q{v} * (Q{v}' * Ps);
        end

        Ps = Ps - eta_Ps * grad_Ps;                  % gradient step

        % SVD projection → Stiefel manifold: Ps^T Ps = I_c
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';                            % m × c
    end

    %% Step 5: Update Q^{(v)} (Sylvester via Kronecker, exact closed-form)
    %   E·Q·F + G·Q = C  →  (F^T ⊗ E + I_c ⊗ G) vec(Q) = vec(C)
    %   where:
    %     E = R_v^T R_v + λ I_m            [m × m]
    %     F = A_v A_v^T                     [c × c]
    %     G = β Ps Ps^T                     [m × m]
    %     C = R_v^T X_v A_v^T + λ S A_v^T - E·Ps·F   [m × c]
    for v = 1:V
        E_Q = R{v}' * R{v} + lambda * eye(m);         % m × m
        F_Q = A{v} * A{v}';                            % c × c
        G_Q = beta * (Ps * Ps');                       % m × m

        % RHS: C^{(v)}
        C_Q = R{v}' * X{v} * A{v}' + lambda * S * A{v}' - E_Q * Ps * F_Q;  % m × c

        % Kronecker solve: (F^T ⊗ E + I ⊗ G) vec(Q) = vec(C)
        K = kron(F_Q', E_Q) + kron(eye(c), G_Q);      % mc × mc
        q_vec = K \ C_Q(:);                             % mc × 1
        Q{v} = reshape(q_vec, [m, c]);                 % m × c
    end

    %% Compute Objective
    obj_rec  = 0;  obj_cons = 0;  obj_psq = 0;
    for v = 1:V
        Mv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
        obj_cons = obj_cons + norm(Mv * A{v} - S, 'fro')^2;
        obj_psq  = obj_psq  + norm(Ps' * Q{v}, 'fro')^2;
    end
    obj = obj_rec + lambda * obj_cons + beta * obj_psq;
    obj_history(iter) = obj;

    %% Convergence
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e\n', ...
                iter, obj, rel_change, obj_rec, lambda*obj_cons, beta*obj_psq);
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
    fprintf('Iter: %d, Obj: %.4e\n', iter, obj);
    fprintf('Recon: %.4e  Cons: %.4e  PsQ: %.4e\n', ...
            obj_rec, lambda*obj_cons, beta*obj_psq);
    fprintf('||Ps^T Ps - I|| = %.2e\n', norm(Ps'*Ps - eye(c), 'fro'));
end
end

%% ==================== Helpers ====================

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
