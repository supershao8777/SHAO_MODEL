function [R, Ps, Q, A, S, obj_history] = bcd_mvrl7(X, m, c, lambda, beta, opts)
% BCD_MVRL7  Multi-View Representation Learning (L21 row-sparse Q).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl7(X, m, c, lambda, beta, opts)
%
%   Objective:
%     min Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%         + λ Σ_v ||Ps A^{(v)} - S||_F^2              ← consensus uses Ps only
%         + β Σ_v ||Q^{(v)}||_{2,1}                   ← row-sparsity on Q
%
%   where ||Q^{(v)}||_{2,1} = Σ_{i=1}^m ||Q^{(v)}_{i,:}||_2
%
%   Constraints:
%     Ps  : m×c,  Ps^T Ps = I_c                           (Stiefel)
%     A{v}: c×n,  A{v} ≥ 0, (A{v})^T 1_c = 1_n          (col-stochastic)
%     S   : m×n,  S ≥ 0, S^T 1_m = 1_n                  (col-stochastic)
%     R{v}, Q{v}: UNCONSTRAINED
%
%   Improvements over baseline:
%     (1) Ps inner loop converges on gradient norm (not fixed steps)
%     (2) Q inner loop converges on subproblem objective (not fixed steps)
%     (3) Q warm-started from Ps reconstruction residual
%
%   Data format: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter     = get_opt(opts, 'max_iter',     50);
max_inner    = get_opt(opts, 'max_inner',    50);   % max inner iters for Ps/Q
tol_inner    = get_opt(opts, 'tol_inner',    1e-4); % inner convergence tolerance
tol_outer    = get_opt(opts, 'tol',          1e-4);
verbose      = get_opt(opts, 'verbose',     true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon = 1e-8;

if verbose
    fprintf('\n=== BCD-MVRL7 Initialization ===\n');
    fprintf('Samples: %d, Proj: %d, Anchors: %d, Views: %d\n', n, m, c, V);
    fprintf('lambda: %.4f, beta: %.4f\n', lambda, beta);
    fprintf('Inner loop: max %d iters, tol %.0e (grad-norm for Ps, obj-change for Q)\n', ...
            max_inner, tol_inner);
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
    [Ps, ~] = qr(randn(m, c), 0);
else
    Ps = randn(m, c) / sqrt(m);
end

% --- S (m × n, column-stochastic) ---
S = col_simplex_project(rand(m, n));

% --- A^{(v)} (c × n, column-stochastic) ---
A = cell(1, V);
for v = 1:V
    A{v} = col_simplex_project(rand(c, n));
end

% --- Q^{(v)}: warm-start from Ps reconstruction residual ---
%   R_v and A_v are random, so first do a quick A update with Ps+0, then
%   capture the residual as initial Q. This prevents Q from being
%   immediately shrunk to zero by L21 before it contributes to the model.
Q = cell(1, V);
for v = 1:V
    % Quick A update assuming Q=0
    RtR = R{v}' * R{v};
    MA_tmp = Ps' * RtR * Ps + lambda * eye(c) + epsilon * eye(c);
    rhs_tmp = Ps' * R{v}' * X{v} + lambda * Ps' * S;
    A_tmp = col_simplex_project(MA_tmp \ rhs_tmp);

    % Residual that Ps alone cannot reconstruct
    recon_ps = R{v} * Ps * A_tmp;                   % n×d_v via d_v×n?
    % Wait — R is d_v×m, Ps is m×c, A is c×n → R*Ps*A is d_v×n ✓
    residual = X{v} - R{v} * Ps * A_tmp;             % d_v × n

    % Project residual to Q-space: min ||residual - R*Q*A||²
    % Approximate: Q_init ≈ pinv(R)*residual*pinv(A)
    Q_init = (R{v}' * R{v} + epsilon * eye(m)) \ (R{v}' * residual * A_tmp');
    Q_init = Q_init / (A_tmp * A_tmp' + epsilon * eye(c));

    Q{v} = Q_init;
end

if verbose
    fprintf('Init done. Q warm-started from Ps residual.\n');
    fprintf('\n  Iter   Objective       RelChg    Recon       Cons       L21(Q)\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------\n');
end

%% ==================== BCD Main Loop ====================
obj_history = zeros(max_iter, 1);

for iter = 1:max_iter
    %% Step 1: Update S (consensus = mean of Ps·A_v, then simplex)
    S_mean = zeros(m, n);
    for v = 1:V
        S_mean = S_mean + Ps * A{v};                 % m × n
    end
    S = col_simplex_project(S_mean / V);

    %% Step 2: Update A^{(v)}  (closed-form, uses Ps^T Ps = I_c)
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        RtR = R{v}' * R{v};                          % m × m

        M_A = Mv' * RtR * Mv + lambda * eye(c) + epsilon * eye(c);  % c × c
        rhs_A = Mv' * R{v}' * X{v} + lambda * Ps' * S;               % c × n
        A{v} = M_A \ rhs_A;
        A{v} = col_simplex_project(A{v});
    end

    %% Step 3: Update R^{(v)} (pseudo-inverse)
    for v = 1:V
        Yv = (Ps + Q{v}) * A{v};                     % m × n
        R{v} = X{v} * Yv' / (Yv * Yv' + epsilon * eye(m));
    end

    %% Step 4: Update Ps (adaptive GD + SVD Stiefel projection)
    %   Iterate until gradient norm converges (not fixed steps)
    S_AA  = zeros(c, c);
    max_W = 0;
    for v = 1:V
        S_AA = S_AA + A{v} * A{v}';
        max_W = max(max_W, norm(R{v}' * R{v}, 2));
    end
    L_Ps = 2 * max_W * norm(S_AA,2) + 2 * lambda * norm(S_AA,2);
    eta_Ps = 1 / max(L_Ps, epsilon);

    % Initial gradient norm for relative tolerance
    grad_norm_init = inf;
    for inner_ps = 1:max_inner
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};
            grad_Ps = grad_Ps + R{v}' * E1 * A{v}';
            E2 = Ps * A{v} - S;
            grad_Ps = grad_Ps + lambda * E2 * A{v}';
        end
        grad_Ps = 2 * grad_Ps;
        gn = norm(grad_Ps, 'fro');

        if inner_ps == 1, grad_norm_init = gn; end

        Ps = Ps - eta_Ps * grad_Ps;
        [U_ps, ~, V_ps] = svd(Ps, 'econ');
        Ps = U_ps * V_ps';

        % Convergence: gradient norm sufficiently reduced
        if gn < tol_inner * grad_norm_init || gn < 1e-10
            break;
        end
    end

    %% Step 5: Update Q^{(v)} (adaptive Proximal Gradient for L21)
    %   Iterate until subproblem objective converges
    for v = 1:V
        L_Qv = 2 * norm(R{v}'*R{v}, 2) * norm(A{v}*A{v}', 2);
        eta_Q = 1 / max(L_Qv, epsilon);

        % Subproblem objective: f(Q) = ||X - R(Ps+Q)A||² + β||Q||_{2,1}
        f_Q_old = inf;

        for inner_q = 1:max_inner
            % Compute subproblem objective (before update)
            Mv_old = Ps + Q{v};
            f_Q = norm(X{v} - R{v} * Mv_old * A{v}, 'fro')^2;
            l21_q = 0;
            for i = 1:m
                l21_q = l21_q + norm(Q{v}(i, :));
            end
            f_Q = f_Q + beta * l21_q;

            % Half-gradient G
            G_Q = R{v}' * (R{v} * Mv_old * A{v} - X{v}) * A{v}';

            % Full gradient step
            Q_tilde = Q{v} - 2 * eta_Q * G_Q;

            % Row-wise group soft-thresholding
            thresh = eta_Q * beta;
            for i = 1:m
                rn = norm(Q_tilde(i, :));
                if rn > thresh
                    Q{v}(i, :) = (1 - thresh / rn) * Q_tilde(i, :);
                else
                    Q{v}(i, :) = 0;
                end
            end

            % Convergence: subproblem objective change
            if inner_q > 1 && abs(f_Q_old - f_Q) / max(abs(f_Q_old), epsilon) < tol_inner
                break;
            end
            f_Q_old = f_Q;
        end
    end

    %% Compute Global Objective
    obj_rec  = 0;  obj_cons = 0;  obj_l21 = 0;
    for v = 1:V
        Mv = Ps + Q{v};
        obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
        obj_cons = obj_cons + norm(Ps * A{v} - S, 'fro')^2;
        for i = 1:m
            obj_l21 = obj_l21 + norm(Q{v}(i, :));
        end
    end
    obj = obj_rec + lambda * obj_cons + beta * obj_l21;
    obj_history(iter) = obj;

    %% Outer Convergence
    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj) / max(abs(obj_history(iter-1)), epsilon);
    end

    if verbose && (mod(iter, 5) == 1 || iter == 1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e\n', ...
                iter, obj, rel_change, obj_rec, lambda*obj_cons, beta*obj_l21);
    end

    if iter > 1 && rel_change < tol_outer
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
    fprintf('Recon: %.4e  Cons: %.4e  L21(Q): %.4e\n', obj_rec, lambda*obj_cons, beta*obj_l21);
    fprintf('||Ps^T Ps-I||=%.2e  Q active rows:', norm(Ps'*Ps - eye(c), 'fro'));
    for v = 1:V
        nr = sum(sqrt(sum(Q{v}.^2, 2)) > epsilon);
        fprintf(' V%d:%d', v, nr);
    end
    fprintf('\n');
end
end

%% ==================== Helpers ====================

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
