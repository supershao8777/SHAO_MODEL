function [R, Ps, Q, A, obj_history] = mvc_ao(X, c, m, alpha, gamma, opts)
% MVC_AO  Multi-View Clustering via Alternating Optimization.
%
%   [R, Ps, Q, A, obj] = mvc_ao(X, c, m, alpha, gamma, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R*(Ps + Q^{(v)})*A^{(v)}||_F^2
%         + α Σ_v ||Q^{(v)}||_F^2 + γ Σ_v ||A^{(v)}||_F^2
%
%   Inputs:
%     X      — cell array {1×V}, X{v} is n × d_v (samples × features)
%     c      — number of clusters
%     m      — number of anchors (c ≤ m recommended)
%     alpha  — regularization for view-specific offset Q
%     gamma  — regularization for anchor dictionary A
%     opts   — optional struct with fields:
%       .max_iter     — max AO outer iterations   (default 50)
%       .pgd_steps    — PGD inner steps for R     (default 5)
%       .cg_max_iter  — max CG iterations          (default 30)
%       .cg_tol       — CG tolerance               (default 1e-5)
%       .tol          — convergence tolerance      (default 1e-4)
%       .verbose      — display progress           (default true)
%
%   Outputs:
%     R      — n × c, shared soft cluster indicator (row-stochastic)
%     Ps     — c × m, shared cluster prototype (orthogonal: Ps*Ps'=I)
%     Q      — {1×V} cell, Q{v} is c × m view-specific offset
%     A      — {1×V} cell, A{v} is m × d_v anchor dictionary
%     obj    — objective value history
%
%   Variables summary:
%     X^{(v)} : n × d_v    Original features (view v)
%     R       : n × c      Shared soft cluster assignment (simplex)
%     Ps      : c × m      Shared cluster→anchor prototype (orthogonal)
%     Q^{(v)} : c × m      View-specific prototype offset (unconstrained)
%     A^{(v)} : m × d_v    Anchor→feature dictionary (unconstrained)
%
%   Reconstruction: X^{(v)} ≈ R * (Ps + Q^{(v)}) * A^{(v)}
%                             n×d_v    n×c    c×m        m×d_v

%% Parse options
if nargin < 6, opts = struct(); end
max_iter    = get_opt(opts, 'max_iter',    50);
pgd_steps   = get_opt(opts, 'pgd_steps',   5);
cg_max_iter = get_opt(opts, 'cg_max_iter', 30);
cg_tol      = get_opt(opts, 'cg_tol',      1e-5);
tol         = get_opt(opts, 'tol',         1e-4);
verbose     = get_opt(opts, 'verbose',     true);

%% Validate dimensions
V = length(X);          % number of views
[n, ~] = size(X{1});    % samples (all views have same n)
assert(c <= m, 'Need c <= m for orthogonality constraint Ps*Ps''=I (c × m matrix).');

% Feature dimensions per view
d = zeros(1, V);
for v = 1:V
    [nv, d(v)] = size(X{v});
    assert(nv == n, 'All views must have same number of samples.');
end

%% Initialization
if verbose
    fprintf('\n=== MVC-AO Initialization ===\n');
    fprintf('Samples: %d, Clusters: %d, Anchors: %d, Views: %d\n', n, c, m, V);
    fprintf('alpha: %.4f, gamma: %.4f\n', alpha, gamma);
end

% --- Initialize R (K-means on concatenated features) ---
X_cat = horzcat(X{:});          % n × Σd_v
rng(42, 'twister');
[label, ~] = kmeans(X_cat, c, 'MaxIter', 100, 'Replicates', 10);
R = full(sparse(1:n, label, 1, n, c));   % one-hot → soft label

% --- Initialize Ps (random + orthogonalize) ---
Ps = randn(c, m);
[U, ~, V_svd] = svd(Ps, 'econ');
Ps = U * V_svd';                % Ps * Ps' = I_c

% --- Initialize Q (zeros: all views start with same prototype) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = zeros(c, m);
end

% --- Initialize A (K-means centers as anchors) ---
A = cell(1, V);
for v = 1:V
    if n >= m
        % Use K-means to get m representative points as initial anchors
        [~, Cv] = kmeans(X{v}, m, 'MaxIter', 50, 'Replicates', 5);
        A{v} = Cv;               % m × d_v
    else
        % If fewer samples than anchors, use all data + random padding
        A{v} = X{v}(randperm(n, min(n,m)), :);
        if n < m
            A{v} = [A{v}; randn(m-n, d(v)) * 0.01];
        end
    end
end

if verbose, fprintf('Initialization complete.\n'); end

%% Alternating Optimization
obj_history = zeros(max_iter, 1);
epsilon = 1e-8;

if verbose
    fprintf('\n  Iter   Objective       RelChg\n');
    fprintf('  ----  --------------  ------\n');
end

for iter = 1:max_iter
    %% Step 1: Update A^{(v)} for each view
    %   A^{(v)} = (H_v^T H_v + γ I)^{-1} H_v^T X^{(v)}
    %   where H_v = R * (Ps + Q^{(v)})  [n × m]
    for v = 1:V
        Hv = R * (Ps + Q{v});                % n × m
        HtH = Hv' * Hv;                      % m × m
        A{v} = (HtH + gamma * eye(m)) \ (Hv' * X{v});  % m × d_v
    end

    %% Step 2: Update Q^{(v)} for each view via CG
    %   Solve: R^T·R·Q·A·A^T + α·Q = R^T·(X - R·Ps·A)·A^T
    %   M = R^T·R  [c × c], S_v = A^{(v)}·(A^{(v)})^T  [m × m]
    M = R' * R;                              % c × c

    for v = 1:V
        Sv = A{v} * A{v}';                   % m × m
        % Residual without Ps term: E_v = X^{(v)} - R·Ps·A^{(v)}
        Ev = X{v} - R * Ps * A{v};           % n × d_v
        B_Q = R' * Ev * A{v}';               % c × m (RHS)

        % Linear operator: L(Q) = M·Q·Sv + α·Q
        L_op = @(Qmat) M * Qmat * Sv + alpha * Qmat;

        % Solve via CG
        Q{v} = matrix_cg(L_op, B_Q, Q{v}, cg_max_iter, cg_tol);
    end

    %% Step 3: Update Ps (shared prototype with orthogonality constraint)
    %   Solve: M·Ps·S_total = G_total, then project to Stiefel manifold.
    %   S_total = Σ_v A^{(v)}·(A^{(v)})^T  [m × m]
    %   G_total = Σ_v R^T·(X^{(v)} - R·Q^{(v)}·A^{(v)})·(A^{(v)})^T  [c × m]
    Stotal = zeros(m, m);
    Gtotal = zeros(c, m);
    for v = 1:V
        Sv = A{v} * A{v}';
        Stotal = Stotal + Sv;
        Ev_Q = X{v} - R * Q{v} * A{v};       % residual without Q term
        Gtotal = Gtotal + R' * Ev_Q * A{v}';
    end

    % Add small ridge for numerical stability
    Stotal_reg = Stotal + epsilon * eye(m);
    M_reg = M + epsilon * eye(c);

    % Linear operator: L(P) = M_reg·P·Stotal_reg
    L_op_ps = @(Pmat) M_reg * Pmat * Stotal_reg;

    % Solve via CG
    Ps_unc = matrix_cg(L_op_ps, Gtotal, Ps, cg_max_iter, cg_tol);

    % Project to Stiefel manifold: Ps·Ps' = I_c
    [U_ps, ~, V_ps] = svd(Ps_unc, 'econ');
    Ps = U_ps * V_ps';

    %% Step 4: Update R (soft cluster indicator, simplex constraint)
    %   PGD: R ← Π_Δ(R - η·∇_R)
    %   ∇_R = 2·R·S_Z - 2·T_Z
    %   S_Z = Σ_v Z_v·Z_v^T  [c × c], Z_v = (Ps+Q^{(v)})·A^{(v)}  [c × d_v]
    %   T_Z = Σ_v X^{(v)}·Z_v^T  [n × c]
    %   Lipschitz constant: L_R = 2·||S_Z||_2

    S_Z = zeros(c, c);
    T_Z = zeros(n, c);
    for v = 1:V
        Zv = (Ps + Q{v}) * A{v};             % c × d_v
        S_Z = S_Z + Zv * Zv';                % c × c
        T_Z = T_Z + X{v} * Zv';              % n × c
    end

    % Step size (inverse of Lipschitz constant)
    L_R = 2 * norm(S_Z, 2);
    eta_R = 1 / max(L_R, epsilon);

    % Projected Gradient Descent (inner iterations)
    for pgd = 1:pgd_steps
        grad_R = 2 * R * S_Z - 2 * T_Z;      % n × c
        R = R - eta_R * grad_R;              % gradient step
        R = project_simplex(R);              % project each row to simplex
    end

    %% Compute objective function
    obj = 0;
    for v = 1:V
        Zv = Ps + Q{v};                       % c × m
        recon = R * Zv * A{v};                % n × d_v
        obj = obj + norm(X{v} - recon, 'fro')^2;
        obj = obj + alpha * norm(Q{v}, 'fro')^2;
        obj = obj + gamma * norm(A{v}, 'fro')^2;
    end
    obj_history(iter) = obj;

    %% Convergence check
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e\n', iter, obj, rel_change);
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

% Final clustering from R
if verbose
    [~, y_pred] = max(R, [], 2);
    fprintf('\n=== Final Summary ===\n');
    fprintf('Iterations: %d,  Objective: %.6e\n', iter, obj);
    fprintf('Cluster sizes from R: ');
    counts = histcounts(y_pred, 1:c+1);
    fprintf('%d ', counts);
    fprintf('\n');
end
end

%% Helper: get option with default
function val = get_opt(opts, field, default)
if isfield(opts, field)
    val = opts.(field);
else
    val = default;
end
end
