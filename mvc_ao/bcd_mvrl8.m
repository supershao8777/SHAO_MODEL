function [R, Ps, Q, A, S, obj_history] = bcd_mvrl8(X, m, c, lambda, beta, opts)
% BCD_MVRL8  Multi-View Representation Learning (IRLS + L21 on Q*A).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl8(X, m, c, lambda, beta, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2       ← consensus on M=Ps+Q
%         + β Σ_v ||Q^{(v)} A^{(v)}||_{2,1}              ← L21 row-sparse on Q*A
%
%   Constraints:
%     R{v}: d_v×m,  R{v} R{v}^T = I_{d_v}       (orthonormal rows, needs d_v ≤ m)
%     Ps  : m×c,    Ps^T Ps = I_c                (Stiefel)
%     A{v}: c×n,    A{v} ≥ 0, col-sum=1          (column-stochastic)
%     S   : m×n,    S ≥ 0, col-sum=1             (column-stochastic)
%     Q{v}: m×c,    UNCONSTRAINED
%
%   Updates (order S→A→R→Ps→Q):
%     S  — simplex mean
%     A  — IRLS closed-form (weighted by D from ||Q*A||_{2,1})
%     R  — Orthogonal Procrustes (SVD)
%     Ps — GD + SVD Stiefel projection
%     Q  — GD with IRLS weights

%% Parse options
if nargin < 6, opts = struct(); end
max_iter   = get_opt(opts, 'max_iter',   50);
pgd_steps  = get_opt(opts, 'pgd_steps',  5);
irls_iters = get_opt(opts, 'irls_iters', 3);    % IRLS inner iterations
tol        = get_opt(opts, 'tol',        1e-4);
verbose    = get_opt(opts, 'verbose',    true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

max_dv = 0; min_dv = inf;
for v = 1:V
    [dv, nv] = size(X{v});
    assert(nv == n, 'All views must have same sample count.');
    max_dv = max(max_dv, dv);
    min_dv = min(min_dv, dv);
end

if verbose
    fprintf('\n=== BCD-MVRL8 Initialization ===\n');
    fprintf('Samples: %d, Proj: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, beta: %.4f\n', lambda, beta);
    fprintf('d_v range: [%d, %d] | m=%d (need d_v≤m for R*R^T=I)\n', min_dv, max_dv, m);
    fprintf('Constraints: R*R^T=I, Ps^T Ps=I, A/S col-stochastic\n');
    fprintf('Regularizer: L21 on Q*A (IRLS with %d inner iters)\n', irls_iters);
end

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, orthonormal rows if d_v ≤ m) ---
R = cell(1, V);
for v = 1:V
    dv = size(X{v}, 1);
    if dv <= m
        % QR of m×d_v gives m×d_v Q, then transpose → d_v×m with orthonormal rows
        [Qr, ~] = qr(randn(m, dv), 0);
        R{v} = Qr';                              % d_v × m, R*R^T = I_{d_v}
    else
        % d_v > m: rows can't be orthonormal, use Procrustes-style init
        [R{v}, ~] = qr(randn(dv, m), 0);         % d_v × m, R^T R = I_m (relaxed)
        if verbose
            fprintf('  Note: View %d (d_v=%d > m=%d), R{%d} rows relaxed.\n', v, dv, m, v);
        end
    end
end

% --- Ps (m × c, orthonormal columns) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);               % Ps^T Ps = I_c
else
    Ps = randn(m, c) / sqrt(m);
    if verbose, fprintf('  Note: m=%d < c=%d, Ps orthogonality relaxed.\n', m, c); end
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
    fprintf('\n  Iter   Objective       RelChg    Recon       Cons       L21(QA)\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------\n');
end

%% ==================== BCD Main Loop ====================
obj_history = zeros(max_iter, 1);

for iter = 1:max_iter
    %% Step 1: Update S (consensus = mean of M·A_v, then simplex)
    %   M_v = Ps + Q_v
    S_mean = zeros(m, n);
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        S_mean = S_mean + Mv * A{v};                 % m × n
    end
    S = col_simplex_project(S_mean / V);

    %% Step 2: Update A^{(v)} (IRLS closed-form + simplex)
    %   D_{ii} = 1/(2||(Q_v·A_v)_{i,:}||₂ + ε)       [m × m diagonal]
    %   A = (M^T R^T R M + λ M^T M + β Q^T D Q)^{-1} (M^T R^T X + λ M^T S)
    %
    %   IRLS: re-compute D from updated A several times
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        RtR = R{v}' * R{v};                          % m × m
        base_M = Mv' * RtR * Mv + lambda * (Mv' * Mv); % c × c (const w.r.t. D)
        base_rhs = Mv' * R{v}' * X{v} + lambda * Mv' * S; % c × n

        for irls = 1:irls_iters
            % Weight matrix D from current Q*A row norms
            QA = Q{v} * A{v};                        % m × n
            d_diag = 1 ./ (2 * sqrt(sum(QA.^2, 2)) + epsilon);  % m × 1
            D = diag(d_diag);                        % m × m

            % IRLS-weighted system
            M_A = base_M + beta * Q{v}' * D * Q{v};  % c × c
            A{v} = M_A \ base_rhs;                   % c × n
            A{v} = col_simplex_project(A{v});
        end
    end

    %% Step 3: Update R^{(v)} (Orthogonal Procrustes)
    %   Y_v = (Ps+Q_v) A_v  [m × n]
    %   [U,~,V] = svd(X_v Y_v^T) → R_v = U V^T
    %   When d_v ≤ m: R*R^T = I_{d_v}; when d_v > m: R^T R = I_m
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                     % m × n
        [U_R, ~, V_R] = svd(X{v} * Yv', 'econ');     % d_v × m
        R{v} = U_R * V_R';                           % d_v × m
    end

    %% Step 4: Update Ps (GD + SVD Stiefel projection)
    %   ∇Ps = 2 Σ_v [R_v^T (R_v M_v A_v - X_v) A_v^T]          (recon)
    %        + 2λ Σ_v [(M_v A_v - S) A_v^T]                      (consensus)
    %
    %   Lipschitz: L = 2 max(||W_v||) ||Σ A_v A_v^T||(1+λ)
    S_AA  = zeros(c, c);
    max_W = 0;
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';
        max_W = max(max_W, norm(R{v}' * R{v}, 2));
    end
    L_Ps = 2 * (max_W + lambda) * norm(S_AA, 2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';   % m × c
            E2 = Mv * A{v} - S;
            grad_Ps = grad_Ps + lambda * E2 * A{v}';   % m × c
        end
        grad_Ps = 2 * grad_Ps;

        Ps = Ps - eta_Ps * grad_Ps;
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';                            % m × c
    end

    %% Step 5: Update Q^{(v)} (GD with IRLS weights)
    %   ∇Qv = 2 R_v^T (R_v M_v A_v - X_v) A_v^T               (recon)
    %        + 2λ (M_v A_v - S) A_v^T                           (consensus)
    %        + 2β D Q_v A_v A_v^T                               (L21 via IRLS)
    %
    %   D from current Q_v*A_v row norms
    for v = 1:V
        Mv = Ps + Q{v};
        L_Qv = 2 * norm(R{v}'*R{v}, 2) * norm(A{v}*A{v}', 2) ...
             + 2 * lambda * norm(A{v}*A{v}', 2);
        eta_Q = 1 / max(L_Qv, epsilon);

        for pgd = 1:pgd_steps
            Mv = Ps + Q{v};

            % IRLS weight matrix from Q*A
            QA = Q{v} * A{v};                        % m × n
            d_diag = 1 ./ (2 * sqrt(sum(QA.^2, 2)) + epsilon);
            D = diag(d_diag);                        % m × m

            % Recon + consensus gradient
            E1 = R{v} * Mv * A{v} - X{v};
            grad_Qv = 2 * R{v}' * E1 * A{v}';        % m × c
            E2 = Mv * A{v} - S;
            grad_Qv = grad_Qv + 2 * lambda * E2 * A{v}';

            % L21 IRLS gradient: 2β D Q A A^T
            grad_Qv = grad_Qv + 2 * beta * D * Q{v} * (A{v} * A{v}');

            % The L21 IRLS term also affects the Lipschitz constant
            % Add its contribution
            L_irls = 2 * beta * norm(D, 2) * norm(A{v}*A{v}', 2);
            eta_eff = 1 / max(L_Qv + L_irls, epsilon);

            Q{v} = Q{v} - eta_eff * grad_Qv;         % m × c
        end
    end

    %% Compute Objective
    obj_rec  = 0;  obj_cons = 0;  obj_l21 = 0;
    for v = 1:V
        Mv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
        obj_cons = obj_cons + norm(Mv * A{v} - S, 'fro')^2;

        % L21 on Q*A: sum of row 2-norms
        QA = Q{v} * A{v};
        for i = 1:m
            obj_l21 = obj_l21 + norm(QA(i, :));
        end
    end
    obj = obj_rec + lambda * obj_cons + beta * obj_l21;
    obj_history(iter) = obj;

    %% Convergence
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e\n', ...
                iter, obj, rel_change, obj_rec, lambda*obj_cons, beta*obj_l21);
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
    fprintf('Recon: %.4e  Cons: %.4e  L21(QA): %.4e\n', obj_rec, lambda*obj_cons, beta*obj_l21);
    % Orthogonality checks
    err_R = 0;
    for v = 1:V
        dv = size(X{v}, 1);
        if dv <= m
            err_R = err_R + norm(R{v}*R{v}' - eye(dv), 'fro')^2;
        else
            err_R = err_R + norm(R{v}'*R{v} - eye(m), 'fro')^2;
        end
    end
    fprintf('||R·R^T-I||_avg=%.2e, ||Ps^T Ps-I||=%.2e\n', sqrt(err_R/V), norm(Ps'*Ps - eye(c),'fro'));
end
end

%% ==================== Helpers ====================

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
