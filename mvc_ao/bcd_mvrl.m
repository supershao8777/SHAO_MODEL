function [R, Ps, Q, A, obj_history] = bcd_mvrl(X, r, m, alpha, gamma, opts)
% BCD_MVRL  Multi-View Representation Learning via Block Coordinate Descent.
%
%   [R, Ps, Q, A, obj] = bcd_mvrl(X, r, m, alpha, gamma, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R*(Ps + Q^{(v)})*A^{(v)}||_F^2
%         + α Σ_{v<u} ||Q^{(v)} ⊙ Q^{(u)}||_1
%         + γ Σ_v ||A^{(v)}||_F^2
%
%   Constraints:
%     R   : n × r,  R^T R = I_r         (orthogonal)
%     Ps  : r × m,  Ps^T Ps = I_m        (orthogonal, requires m ≤ r)
%     Q{v}: r × m,  Q{v} ≥ 0            (non-negative)
%     A{v}: m × d_v, A{v} ≥ 0           (non-negative)
%
%   Inputs:
%     X      — cell array {1×V}, X{v} is n × d_v (samples × features)
%     r      — latent dimension (r ≥ m required)
%     m      — number of prototypes
%     alpha  — redundancy penalty (L1 on cross-view Q products)
%     gamma  — ridge penalty on A
%     opts   — optional struct:
%       .max_iter      — max outer BCD iterations (default 50)
%       .pgd_steps     — PGD inner steps for Ps/Q (default 1)
%       .tol           — convergence tolerance       (default 1e-4)
%       .verbose       — display progress            (default true)
%
%   Outputs:
%     R      — n × r,       shared representation (orthogonal columns)
%     Ps     — r × m,       shared prototype (orthogonal columns)
%     Q      — {1×V} cell,  Q{v} is r × m, ≥0
%     A      — {1×V} cell,  A{v} is m × d_v, ≥0
%     obj    — objective value history
%
%   Variables summary:
%     X^{(v)} : n × d_v    Original features (view v)
%     R       : n × r      Shared representation     [R^T R = I_r]
%     Ps      : r × m      Shared prototype          [Ps^T Ps = I_m]
%     Q^{(v)} : r × m      View-specific offset      [Q ≥ 0]
%     A^{(v)} : m × d_v    Anchor dictionary          [A ≥ 0]
%
%   Reconstruction: X^{(v)} ≈ R * (Ps + Q^{(v)}) * A^{(v)}
%                             n×d_v   n×r   r×m        m×d_v

%% Parse options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 1);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

%% Validate dimensions
V = length(X);                              % number of views
[n, ~] = size(X{1});                        % samples
assert(m <= r, 'Need m ≤ r for orthogonality constraint Ps^T Ps = I_m (Ps is r×m).');

d = zeros(1, V);
for v = 1:V
    [nv, d(v)] = size(X{v});
    assert(nv == n, 'All views must have same number of samples.');
end

%% Initialization
if verbose
    fprintf('\n=== BCD-MVRL Initialization ===\n');
    fprintf('Samples: %d, Latent: %d, Prototypes: %d, Views: %d\n', n, r, m, V);
    fprintf('alpha: %.4f, gamma: %.4f\n', alpha, gamma);
end

% --- Initialize R (random orthogonal) ---
rng(42, 'twister');
[R_init, ~] = qr(randn(n, r), 0);           % n × r, R^T R = I_r
R = R_init;

% --- Initialize Ps (random orthogonal, r × m) ---
[Ps_init, ~] = qr(randn(r, m), 0);          % r × m, Ps^T Ps = I_m
Ps = Ps_init;

% --- Initialize Q (small non-negative values) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = max(rand(r, m) * 0.1 - 0.05, 0);  % r × m, small non-negative
end

% --- Initialize A (non-negative, from K-means of X^T) ---
A = cell(1, V);
for v = 1:V
    % Cluster the features (columns of X) to get m prototype features
    if d(v) >= m
        [~, Cv] = kmeans(X{v}', m, 'MaxIter', 100, 'Replicates', 3);
        A{v} = max(Cv, 0);                   % m × d_v, ≥0
    else
        A{v} = max(rand(m, d(v)), 0) * 0.01;
    end
end

if verbose, fprintf('Initialization complete.\n'); end

%% BCD Optimization
obj_history = zeros(max_iter, 1);
epsilon = 1e-8;

if verbose
    fprintf('\n  Iter   Objective       RelChg\n');
    fprintf('  ----  --------------  ------\n');
end

for iter = 1:max_iter
    %% Step 1: Update A^{(v)} (Ridge Regression + non-negativity)
    %   M^{(v)} = R*(Ps + Q^{(v)})  [n × m]
    %   A^{(v)} = max( (M^T M + γI)^{-1} M^T X^{(v)},  0 )
    for v = 1:V
        Mv = R * (Ps + Q{v});                  % n × m
        MtM = Mv' * Mv;                        % m × m
        A{v} = (MtM + gamma * eye(m)) \ (Mv' * X{v});   % m × d_v
        A{v} = max(A{v}, 0);                   % enforce non-negativity
    end

    %% Step 2: Update Ps (Projected Gradient + SVD)
    %   ∇Ps = 2 R^T Σ_v (R(Ps+Qv)Av - Xv) Av^T
    %   Ps ← U V^T  after [U,~,V] = svd(Ps - η·∇Ps)
    %
    %   Lipschitz: L_Ps = 2 ||Σ_v Av Av^T||_2

    Stotal = zeros(m, m);
    for v = 1:V
        Stotal = Stotal + A{v} * A{v}';         % m × m
    end
    L_Ps = 2 * norm(Stotal, 2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    for pgd = 1:pgd_steps
        grad_Ps = 0;
        for v = 1:V
            Ev = R * (Ps + Q{v}) * A{v} - X{v};  % n × d_v
            grad_Ps = grad_Ps + R' * Ev * A{v}'; % r × m
        end
        grad_Ps = 2 * grad_Ps;                   % r × m

        Ps = Ps - eta_Ps * grad_Ps;              % gradient step

        % SVD projection to Stiefel: Ps^T Ps = I_m
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';                       % r × m
    end

    %% Step 3: Update Q^{(v)} (Proximal Gradient: L1 cross-view + non-negativity)
    %   ∇Qv = 2 R^T (R(Ps+Qv)Av - Xv) Av^T
    %   G = Qv - η·∇Qv
    %   Qv = max( G - η·α·Σ_{u≠v}|Qu|,  0 )    [Soft + non-negativity]
    %
    %   Lipschitz: L_Qv = 2 ||Av Av^T||_2

    for v = 1:V
        L_Qv = 2 * norm(A{v} * A{v}', 2);
        eta_Q = 1 / max(L_Qv, epsilon);

        for pgd = 1:pgd_steps
            Ev = R * (Ps + Q{v}) * A{v} - X{v};  % n × d_v
            grad_Qv = 2 * R' * Ev * A{v}';        % r × m

            G = Q{v} - eta_Q * grad_Qv;            % gradient step

            % Cross-view L1 threshold (element-wise)
            sum_other_Q = zeros(r, m);
            for u = 1:V
                if u ~= v
                    sum_other_Q = sum_other_Q + abs(Q{u});
                end
            end
            tau_Q = eta_Q * alpha * sum_other_Q;   % r × m threshold

            % Soft-threshold + non-negativity
            Q{v} = max(G - tau_Q, 0);              % r × m, ≥0
        end
    end

    %% Step 4: Update R (Orthogonal Procrustes)
    %   B^{(v)} = (Ps + Q^{(v)}) A^{(v)}  [r × d_v]
    %   M = Σ_v X^{(v)} (B^{(v)})^T       [n × r]
    %   [U,~,V] = svd(M, 'econ');  R = U V^T

    M_sum = zeros(n, r);
    for v = 1:V
        Bv = (Ps + Q{v}) * A{v};                 % r × d_v
        M_sum = M_sum + X{v} * Bv';              % n × r
    end

    [U_R, ~, V_R] = svd(M_sum, 'econ');
    R = U_R * V_R';                              % n × r, R^T R = I_r

    %% Compute Objective
    obj = 0;
    obj_recon = 0;
    obj_l1 = 0;

    for v = 1:V
        recon = R * (Ps + Q{v}) * A{v};          % n × d_v
        obj_recon = obj_recon + norm(X{v} - recon, 'fro')^2;
        obj = obj + gamma * norm(A{v}, 'fro')^2;
    end

    % Cross-view L1 redundancy penalty
    for v = 1:V
        for u = v+1:V
            obj_l1 = obj_l1 + sum(sum(abs(Q{v} .* Q{u})));
        end
    end

    obj = obj_recon + alpha * obj_l1 + obj;
    obj_history(iter) = obj;

    %% Convergence check
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  (recon=%.4e L1=%.4e)\n', ...
                iter, obj, rel_change, obj_recon, alpha*obj_l1);
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
    [~, y_pred] = max(R, [], 2);
    fprintf('\n=== Final Summary ===\n');
    fprintf('Iterations: %d,  Objective: %.6e\n', iter, obj);
    fprintf('Reconstruction: %.4e,  L1(Q): %.4e\n', obj_recon, alpha*obj_l1);
    % Check orthogonality
    fprintf('||R^T R - I|| = %.2e\n', norm(R'*R - eye(r), 'fro'));
    fprintf('||Ps^T Ps - I|| = %.2e\n', norm(Ps'*Ps - eye(m), 'fro'));
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
