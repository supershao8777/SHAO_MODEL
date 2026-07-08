function [R, Ps, Q, A, S, obj_history] = bcd_mvrl5(X, m, c, lambda, beta, opts)
% BCD_MVRL5  Multi-View Representation Learning (no orthogonality constraints).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl5(X, m, c, lambda, beta, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%         + β Σ_v ||Ps^T Q^{(v)}||_F^2
%
%   Constraints (ONLY these — no orthogonality):
%     A{v}: c×n,  A{v} ≥ 0, (A{v})^T 1_c = 1_n  (column-stochastic)
%     S   : m×n,  S ≥ 0, S^T 1_m = 1_n            (col-stochastic consensus)
%
%   All other variables are UNCONSTRAINED:
%     R{v}: d_v×m,  Ps: m×c,  Q{v}: m×c
%
%   Key features:
%     - λ ||Ps^T Q_v||_F^2 soft penalty replaces hard Q^T Ps = 0 constraint
%     - No γ ridge on A (simplex projection provides regularization)
%     - R via pseudo-inverse, Ps/Q via gradient descent
%
%   Data format: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 3);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

if verbose
    fprintf('\n=== BCD-MVRL5 Initialization ===\n');
    fprintf('Samples: %d, Proj: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, beta: %.4f\n', lambda, beta);
    fprintf('Constraints: A≥0(col-sum=1), S≥0(col-sum=1) — all others unconstrained\n');
end

for v = 1:V
    [dv, nv] = size(X{v});
    assert(nv == n, 'All views must have same sample count.');
end

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, random small) ---
R = cell(1, V);
for v = 1:V
    dv = size(X{v}, 1);
    R{v} = randn(dv, m) * 0.01;
end

% --- Ps (m × c, random small) ---
Ps = randn(m, c) * 0.01;

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

    %% Step 2: Update A^{(v)} (closed-form + simplex, no γ)
    %   M = Ps+Q_v  [m×c]
    %   A = (M^T R^T R M + λ M^T M + εI)^{-1} (M^T R^T X + λ M^T S)
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        RtR = R{v}' * R{v};                          % m × m

        M_A = Mv' * RtR * Mv + lambda * (Mv' * Mv) + epsilon * eye(c);  % c × c
        rhs_A = Mv' * R{v}' * X{v} + lambda * Mv' * S;                   % c × n
        A{v} = M_A \ rhs_A;                                               % c × n

        A{v} = col_simplex_project(A{v});            % ≥0, col-sum=1
    end

    %% Step 3: Update R^{(v)} (unconstrained pseudo-inverse)
    %   Y_v = (Ps+Q_v) A_v  [m × n]
    %   R_v = X_v Y_v^T (Y_v Y_v^T)^{-1}
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                     % m × n
        R{v} = X{v} * Yv' / (Yv * Yv' + epsilon * eye(m));  % d_v × m
    end

    %% Step 4: Update Ps (gradient descent, UNCONSTRAINED — no SVD)
    %   ∇Ps = 2 Σ_v [R_v^T(R_v M_v A_v - X_v)A_v^T]
    %        + 2λ Σ_v [(M_v A_v - S)A_v^T]
    %        + 2λ Σ_v [Q_v Q_v^T Ps]              ← new: ||Ps^T Q_v||² penalty
    %
    %   Lipschitz: L_Ps ≈ 2 max(||W_v||) ||Σ S_v|| + 2λ ||Σ S_v|| + 2λ ||Σ Q_v Q_v^T||

    % Build pre-computed terms
    S_AA = zeros(c, c);     % Σ A_v A_v^T
    H_QQ = zeros(m, m);     % Σ Q_v Q_v^T
    max_W_norm = 0;
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';
        H_QQ = H_QQ + Q{v} * Q{v}';
        max_W_norm = max(max_W_norm, norm(R{v}' * R{v}, 2));
    end
    L_Ps = 2 * max_W_norm * norm(S_AA, 2) + 2 * lambda * norm(S_AA, 2) ...
         + 2 * beta * norm(H_QQ, 2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            % Reconstruction term
            E1 = R{v} * Mv * A{v} - X{v};           % d_v × n
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';  % m × c
            % Consensus term
            E2 = Mv * A{v} - S;                      % m × n
            grad_Ps = grad_Ps + lambda * E2 * A{v}'; % m × c
        end
        grad_Ps = 2 * grad_Ps;

        % β ||Ps^T Q_v||² penalty gradient: 2 Q_v Q_v^T Ps per view
        for v = 1:V
            grad_Ps = grad_Ps + 2 * beta * Q{v} * (Q{v}' * Ps);
        end

        Ps = Ps - eta_Ps * grad_Ps;                 % m × c (unconstrained)
    end

    %% Step 5: Update Q^{(v)} (gradient descent, UNCONSTRAINED)
    %   ∇Qv = 2 R_v^T(R_v M_v A_v - X_v)A_v^T
    %        + 2λ (M_v A_v - S)A_v^T
    %        + 2λ P_s P_s^T Q_v                    ← new: ||Ps^T Q_v||² penalty
    for v = 1:V
        Wv = R{v}' * R{v};                           % m × m
        Sv = A{v} * A{v}';                           % c × c

        L_Qv = 2 * norm(Wv, 2) * norm(Sv, 2) ...
             + 2 * lambda * norm(Sv, 2) ...
             + 2 * beta * norm(Ps * Ps', 2);
        eta_Qv = 1 / max(L_Qv, epsilon);

        for pgd = 1:pgd_steps
            Mv = Ps + Q{v};

            % Recon + consensus gradient
            E1 = R{v} * Mv * A{v} - X{v};
            grad_Qv = 2 * R{v}' * E1 * A{v}';        % m × c
            E2 = Mv * A{v} - S;
            grad_Qv = grad_Qv + 2 * lambda * E2 * A{v}';

            % β ||Ps^T Q_v||² penalty gradient: 2 Ps Ps^T Q_v
            grad_Qv = grad_Qv + 2 * beta * Ps * (Ps' * Q{v});

            Q{v} = Q{v} - eta_Qv * grad_Qv;          % m × c (unconstrained)
        end
    end

    %% Compute Objective
    obj_rec  = 0;   % reconstruction
    obj_cons = 0;   % consensus
    obj_psq  = 0;   % ||Ps^T Q_v||² penalty

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
end
end

%% ==================== Helpers ====================

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
