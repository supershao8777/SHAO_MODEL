function [R, Ps, Q, A, S, obj_history] = bcd_mvrl2(X, m, c, lambda, gamma, opts)
% BCD_MVRL2  Multi-View Representation Learning with Consensus.
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl2(X, m, c, lambda, gamma, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%         + γ Σ_v ||A^{(v)}||_F^2
%
%   Constraints:
%     R{v}: d_v×m,  R{v}^T R{v} = I_m           (orthogonal projection)
%     Ps  : m×c,    Ps^T Ps = I_c                (orthogonal prototype)
%     Q{v}: m×c,    Q{v}^T Ps = 0                (orthogonal complement)
%     A{v}: c×n,    A{v} ≥ 0, A{v}^T 1_c = 1_n  (column-stochastic)
%     S   : m×n,    S ≥ 0, S^T 1_m = 1_n         (consensus, column-stochastic)
%
%   Data format: X{v} is d_v × n  (features × samples)
%
%   Inputs:
%     X      — cell array {1×V}, X{v} is d_v × n
%     m      — projection dimension (m ≤ d_v, m ≥ c required)
%     c      — number of anchors / clusters
%     lambda — consensus alignment weight
%     gamma  — ridge penalty on A
%     opts   — optional struct:
%       .max_iter      — max outer BCD iterations  (default 50)
%       .pgd_steps     — PGD inner steps for Ps/Q   (default 3)
%       .tol           — convergence tolerance      (default 1e-4)
%       .verbose       — display progress           (default true)
%
%   Outputs:
%     R      — {1×V} cell,  R{v} is d_v × m
%     Ps     — m × c,       shared prototype (orthogonal columns)
%     Q      — {1×V} cell,  Q{v} is m × c
%     A      — {1×V} cell,  A{v} is c × n (column-stochastic)
%     S      — m × n,       consensus representation (column-stochastic)
%     obj    — objective value history
%
%   Variables summary:
%     X^{(v)} : d_v × n     Original features (features × samples)
%     R^{(v)} : d_v × m     View-specific projection    [R^T R = I_m]
%     Ps      : m × c       Shared prototype            [Ps^T Ps = I_c]
%     Q^{(v)} : m × c       View-specific offset        [Q^T Ps = 0]
%     A^{(v)} : c × n       Anchor coefficients          [≥0, col-sum=1]
%     S       : m × n       Consensus representation    [≥0, col-sum=1]

%% Parse options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 3);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

%% Validate dimensions
V = length(X);                               % number of views
[d1, n] = size(X{1});                        % features × samples (NOTE: transposed)
if m < c && verbose
    fprintf('  Note: m(=%.0f) < c(=%.0f), Ps orthogonality relaxed (PS^T PS ≈ I will not hold).\n', m, c);
end

for v = 1:V
    [dv, nv] = size(X{v});
    assert(nv == n, 'All views must have same number of samples.');
    if dv < m && verbose
        fprintf('  Note: View %d (d_v=%d < m=%d), R{%d} orthogonality relaxed.\n', v, dv, m, v);
    end
end

%% Initialization
if verbose
    fprintf('\n=== BCD-MVRL2 Initialization ===\n');
    fprintf('Samples: %d, Proj-dim: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, gamma: %.4f\n', lambda, gamma);
end

% --- Initialize R^{(v)} (random orthogonal, d_v × m) ---
R = cell(1, V);
rng(42, 'twister');
for v = 1:V
    dv = size(X{v}, 1);
    if dv >= m
        [R{v}, ~] = qr(randn(dv, m), 0);        % d_v × m, orthonormal columns
    else
        % d_v < m: columns can't be orthonormal, use scaled random
        R{v} = randn(dv, m) / sqrt(dv);          % d_v × m, approx unit-norm columns
    end
end

% --- Initialize Ps (orthogonal if m≥c, else scaled random m×c) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);               % m × c, Ps^T Ps = I_c
else
    Ps = randn(m, c) / sqrt(m);                  % m × c, approx unit-norm cols
end

% --- Initialize Q^{(v)} (small random, then project to null(Ps^T)) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = randn(m, c) * 0.01;                    % m × c
    % Project to null space: Q ← Q - Ps*(Ps^T*Q)
    Q{v} = Q{v} - Ps * (Ps' * Q{v});              % ensures Q^T Ps = 0
end

% --- Initialize S (consensus, from first view) ---
S = max(rand(m, n) * 0.1, 0);                     % m × n, ≥0
S = col_simplex_project(S);                       % column-stochastic

% --- Initialize A^{(v)} (column-stochastic, c × n) ---
A = cell(1, V);
for v = 1:V
    A{v} = rand(c, n);
    A{v} = col_simplex_project(A{v});              % c × n, column-stochastic
end

if verbose, fprintf('Initialization complete.\n'); end

%% BCD Optimization
obj_history = zeros(max_iter, 1);
epsilon = 1e-8;

if verbose
    fprintf('\n  Iter   Objective       RelChg    Recon        Consensus\n');
    fprintf('  ----  --------------  ------   ----------   ----------\n');
end

for iter = 1:max_iter
    %% Step 1: Update S (consensus representation)
    %   S_mean = (1/V) Σ_v (Ps + Q^{(v)}) A^{(v)}
    %   S = simplex-project each column
    S_mean = zeros(m, n);
    for v = 1:V
        S_mean = S_mean + (Ps + Q{v}) * A{v};     % (m×c) × (c×n) = m×n
    end
    S_mean = S_mean / V;
    S = col_simplex_project(S_mean);               % ≥0, col-sum=1

    %% Step 2: Update A^{(v)} (Ridge regression + column-simplex projection)
    %   B^{(v)} = R^{(v)} (Ps+Q^{(v)})           [d_v × c]
    %   C^{(v)} = Ps + Q^{(v)}                    [m × c]
    %   A = (B^T B + λ C^T C + γ I)^{-1} (B^T X + λ C^T S)
    for v = 1:V
        Zv = Ps + Q{v};                            % m × c
        Bv = R{v} * Zv;                            % d_v × c

        % Closed-form: (c × c) system
        M_A = Bv' * Bv + lambda * (Zv' * Zv) + gamma * eye(c);  % c × c
        rhs_A = Bv' * X{v} + lambda * Zv' * S;                    % c × n
        A{v} = M_A \ rhs_A;                                        % c × n

        % Project to column-stochastic simplex
        A{v} = col_simplex_project(A{v});          % ≥0, col-sum=1
    end

    %% Step 3: Update Ps (gradient descent + Stiefel projection)
    %   ∇Ps = Σ_v [R{v}^T (R{v}(Ps+Q{v})A{v} - X{v}) A{v}^T]
    %        + λ Σ_v [((Ps+Q{v})A{v} - S) A{v}^T]
    %   Lipschitz: L = (1+λ) ||Σ_v A{v} A{v}^T||_2
    S_AA = zeros(c, c);
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';                % c × c
    end
    L_Ps = (1 + lambda) * norm(S_AA, 2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Zv = Ps + Q{v};                        % m × c
            E1 = R{v} * Zv * A{v} - X{v};         % d_v × n
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';  % m × c

            E2 = Zv * A{v} - S;                     % m × n
            grad_Ps = grad_Ps + lambda * E2 * A{v}'; % m × c
        end

        Ps = Ps - eta_Ps * grad_Ps;                 % gradient step

        % Project to Stiefel: Ps^T Ps = I_c
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';                          % m × c
    end

    %% Step 4: Update Q^{(v)} (gradient descent + null-space projection)
    %   ∇Qv = R{v}^T (R{v}(Ps+Q{v})A{v} - X{v}) A{v}^T
    %        + λ ((Ps+Q{v})A{v} - S) A{v}^T
    %   Project: Q ← Q - Ps*(Ps^T*Q)  (enforce Q^T Ps = 0)
    for v = 1:V
        L_Qv = (1 + lambda) * norm(A{v} * A{v}', 2);
        eta_Qv = 1 / max(L_Qv, epsilon);

        for pgd = 1:pgd_steps
            Zv = Ps + Q{v};
            E1 = R{v} * Zv * A{v} - X{v};           % d_v × n
            grad_Qv = R{v}' * E1 * A{v}';            % m × c

            E2 = Zv * A{v} - S;                      % m × n
            grad_Qv = grad_Qv + lambda * E2 * A{v}'; % m × c

            Q{v} = Q{v} - eta_Qv * grad_Qv;          % gradient step

            % Project to null space of Ps^T: Q^T Ps = 0
            Q{v} = Q{v} - Ps * (Ps' * Q{v});         % m × c
        end
    end

    %% Step 5: Update R^{(v)} (Orthogonal Procrustes per view)
    %   M^{(v)} = (Ps + Q^{(v)}) A^{(v)}  [m × n]
    %   [U,~,V] = svd(X^{(v)} M^{(v)T})  →  R^{(v)} = U V^T
    for v = 1:V
        Mv = (Ps + Q{v}) * A{v};                    % m × n
        [U_R, ~, V_R] = svd(X{v} * Mv', 'econ');    % d_v × m
        R{v} = U_R * V_R';                          % d_v × m, R^T R = I_m
    end

    %% Compute Objective
    obj_rec = 0;      % reconstruction term
    obj_cons = 0;     % consensus term
    obj_regA = 0;     % ridge on A

    for v = 1:V
        Zv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Zv * A{v}, 'fro')^2;
        obj_cons = obj_cons + norm(Zv * A{v} - S, 'fro')^2;
        obj_regA = obj_regA + norm(A{v}, 'fro')^2;
    end

    obj = obj_rec + lambda * obj_cons + gamma * obj_regA;
    obj_history(iter) = obj;

    %% Convergence check
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %10.2e\n', ...
                iter, obj, rel_change, obj_rec, lambda*obj_cons);
    end

    if iter > 1 && rel_change < tol
        if verbose
            fprintf('  Converged at iteration %d (rel_change < %.0e).\n', iter, tol);
        end
        obj_history = obj_history(1:iter);
        break;
    end
end

if verbose && iter >= max_iter
    fprintf('  Reached max iterations (%d).\n', max_iter);
end

%% Final summary
if verbose
    fprintf('\n=== Final Summary ===\n');
    fprintf('Iterations: %d,  Objective: %.6e\n', iter, obj);
    fprintf('Reconstruction: %.4e,  Consensus: %.4e,  Ridge(A): %.4e\n', ...
            obj_rec, lambda*obj_cons, gamma*obj_regA);

    % Orthogonality checks
    err_R = 0; err_Q = 0;
    valid_R = 0;  % count views where d_v ≥ m (orthogonality expected)
    for v = 1:V
        err_R = err_R + norm(R{v}'*R{v} - eye(m), 'fro')^2;
        err_Q = err_Q + norm(Q{v}'*Ps, 'fro')^2;
        if size(X{v}, 1) >= m, valid_R = valid_R + 1; end
    end
    if valid_R > 0
        fprintf('||R^T R - I||_avg = %.2e  (%d/%d views)\n', sqrt(err_R/V), valid_R, V);
    end
    fprintf('||Ps^T Ps - I||    = %.2e\n', norm(Ps'*Ps - eye(c), 'fro'));
    fprintf('||Q^T Ps||_avg     = %.2e\n', sqrt(err_Q/V));
end
end

%% ==================== Helper Functions ====================

function S_out = col_simplex_project(S_in)
% COL_SIMPLEX_PROJECT  Project each column of S_in to the probability simplex.
%   S_out = col_simplex_project(S_in)
%   After projection: S ≥ 0, sum(S, 1) = 1 (column-stochastic)
%
%   S_in  : m × n
%   S_out : m × n, each column on the (m-1)-simplex

% Use row-wise projection on transposed matrix
S_out = project_simplex(S_in')';   % n×m → project rows → m×n
end

function val = get_opt(opts, field, default)
if isfield(opts, field)
    val = opts.(field);
else
    val = default;
end
end
