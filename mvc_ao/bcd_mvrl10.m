function [R, Ps, Q, A, S, obj_history, diag] = bcd_mvrl10(X, m, c, lambda, beta, opts)
% BCD_MVRL10  多视图表示学习 (Cayley-Ps + NMF-Q + PGD-A)
%
%   [R, Ps, Q, A, S, obj, diag] = bcd_mvrl10(X, m, c, lambda, beta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%       + β Σ_v ||Q^{(v)} A^{(v)}||_F^2
%
%   Constraints:
%     Ps  : m×c,  Ps^T Ps = I_c               (Stiefel, Cayley retraction)
%     Q{v}: m×c,  Q{v} ≥ 0                    (NMF multiplicative update)
%     A{v}: c×n,  A{v} ≥ 0, col-sum=1        (PGD + simplex)
%     S   : m×n,  S ≥ 0, col-sum=1           (simplex mean)
%     R{v}: d_v×m, unconstrained              (pseudo-inverse)
%
%   Update order: R → Ps (Cayley) → Q (NMF) → A (PGD) → S
%
%   Data format: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  100);
pgd_steps = get_opt(opts, 'pgd_steps', 8);
eta_cayley = get_opt(opts, 'eta_cayley', 1e-3);  % Cayley step size
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
epsilon_base = 1e-8;

if verbose
    fprintf('\n========== BCD-MVRL10 (Cayley + NMF + PGD) ==========\n');
    fprintf('Samples: %d, m: %d, c: %d, Views: %d\n', n, m, c, V);
    fprintf('λ=%.4f, β=%.4f | Cayley η=%.1e | PGD steps=%d\n', lambda, beta, eta_cayley, pgd_steps);
    fprintf('Update: R(pinv) → Ps(Cayley) → Q(NMF) → A(PGD) → S(simplex)\n');
end

for v = 1:V
    assert(size(X{v},2) == n, 'All views must have same sample count.');
end

%% Helper: pos/neg split
pos = @(M) (abs(M) + M) / 2;
neg = @(M) (abs(M) - M) / 2;

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, random small) ---
R = cell(1, V);
for v = 1:V
    R{v} = randn(size(X{v},1), m) * 0.01;
end

% --- Ps (m × c, orthogonal) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);
else
    Ps = randn(m, c) / sqrt(m);
end

% --- Q^{(v)} (m × c, non-negative) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = max(rand(m, c) * 0.1, 0);
end

% --- A^{(v)} (c × n, column-stochastic) ---
A = cell(1, V);
for v = 1:V
    A{v} = col_simplex_project(rand(c, n));
end

% --- S (m × n, column-stochastic) ---
S = col_simplex_project(rand(m, n));

% Initial objective
[obj_old, rec0, cons0, qa0] = calc_obj(X, R, Ps, Q, A, S, lambda, beta);
obj_history = zeros(max_iter, 1);
obj_history(1) = obj_old;

if verbose
    fprintf('Init obj: %.4e (recon=%.2e, cons=%.2e, QA=%.2e)\n', obj_old, rec0, lambda*cons0, beta*qa0);
    fprintf('\n  Iter   Objective       RelChg    Recon       Cons       QA         ||Ps^TPs-I||\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------  -----------\n');
end

%% ==================== BCD Main Loop ====================
for iter = 1:max_iter
    %% --- Adaptive ridge from current scale ---
    eps_use = max(epsilon_base, 1e-6 * max(cellfun(@(Av) trace(Av*Av')/c, A)));

    %% Step 1: Update R^{(v)} (pseudo-inverse)
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        Zv = Mv * A{v};                              % m × n
        R{v} = X{v} * Zv' / (Zv * Zv' + 1e-6 * eye(m));  % d_v × m
    end

    %% Step 2: Update Ps (Cayley transform on Stiefel manifold)
    %   G = Σ_v [R_v^T (R_v M_v A_v - X_v) A_v^T + λ (M_v A_v - S) A_v^T]
    G = zeros(m, c);
    for v = 1:V
        Mv = Ps + Q{v};
        E1 = R{v} * Mv * A{v} - X{v};
        G = G + R{v}' * E1 * A{v}';                  % recon gradient
        E2 = Mv * A{v} - S;
        G = G + lambda * E2 * A{v}';                  % consensus gradient
    end
    G = 2 * G;                                       % full gradient

    % Skew-symmetric: W = G*Ps^T - Ps*G^T
    W = G * Ps' - Ps * G';                           % m × m, skew-symmetric

    % Cayley retraction: Ps_new = (I+ηW/2)⁻¹ (I-ηW/2) Ps
    I_m = eye(m);
    C_mat = (I_m + (eta_cayley/2) * W) \ (I_m - (eta_cayley/2) * W);
    Ps = C_mat * Ps;                                  % m × c, orthogonality preserved

    %% Step 3: Update Q^{(v)} (NMF multiplicative update, guarantees Q ≥ 0)
    %   W1 = R_v^T R_v,  W2 = R_v^T X_v A_v^T
    %   Q ← Q ⊙ N ./ (D + ε)
    for v = 1:V
        W1 = R{v}' * R{v};                           % m × m
        W2 = R{v}' * X{v} * A{v}';                   % m × c
        AAT = A{v} * A{v}';                          % c × c

        % Numerator N
        N_Q = neg(W1) * Q{v} * AAT ...
            + neg(W1) * pos(Ps) * AAT ...
            + pos(W1) * neg(Ps) * AAT ...
            + pos(W2) ...
            + lambda * S * A{v}' ...
            + lambda * neg(Ps) * AAT;

        % Denominator D
        D_Q = pos(W1) * Q{v} * AAT ...
            + pos(W1) * pos(Ps) * AAT ...
            + neg(W1) * neg(Ps) * AAT ...
            + neg(W2) ...
            + lambda * pos(Ps) * AAT ...
            + (lambda + beta) * Q{v} * AAT;

        % Multiplicative update
        Q{v} = Q{v} .* N_Q ./ (D_Q + eps_use);
        Q{v} = max(Q{v}, 0);                         % safety clip
    end

    %% Step 4: Update A^{(v)} (Projected Gradient Descent + simplex)
    %   ∇A = 2 M^T R^T (R M A - X) + 2λ M^T (M A - S) + 2β Q^T Q A
    for v = 1:V
        Mv = Ps + Q{v};                              % m × c
        RtR = R{v}' * R{v};                          % m × m

        % Lipschitz step
        L_A = 2 * norm(Mv'*RtR*Mv + lambda*(Mv'*Mv) + beta*(Q{v}'*Q{v}), 2);
        eta_A = 1 / max(L_A, epsilon_base);

        for pgd = 1:pgd_steps
            Mv = Ps + Q{v};
            E1 = R{v} * Mv * A{v} - X{v};
            E2 = Mv * A{v} - S;
            grad_A = 2 * Mv' * R{v}' * E1 ...
                   + 2*lambda * Mv' * E2 ...
                   + 2*beta * Q{v}' * Q{v} * A{v};

            A{v} = A{v} - eta_A * grad_A;
            A{v} = col_simplex_project(A{v});
        end
    end

    %% Step 5: Update S (mean + column-simplex)
    S_mean = zeros(m, n);
    for v = 1:V
        S_mean = S_mean + (Ps + Q{v}) * A{v};        % m × n
    end
    S = col_simplex_project(S_mean / V);

    %% Compute objective & check convergence
    [obj_now, obj_rec, obj_cons, obj_qa] = calc_obj(X, R, Ps, Q, A, S, lambda, beta);
    obj_history(iter) = obj_now;

    rel_change = 0;
    if iter > 1
        rel_change = abs(obj_history(iter-1) - obj_now) / max(abs(obj_history(iter-1)), epsilon_base);
    end

    if verbose && (mod(iter,5)==1 || iter==1)
        ps_err = norm(Ps'*Ps - eye(c), 'fro');
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e  %11.2e\n', ...
                iter, obj_now, rel_change, obj_rec, lambda*obj_cons, beta*obj_qa, ps_err);
    end

    if iter > 1 && rel_change < tol
        if verbose
            fprintf('  Converged at iter %d (rel<%.0e).\n', iter, tol);
        end
        obj_history = obj_history(1:iter);
        break;
    end
end

if verbose && iter >= max_iter
    fprintf('  Max iters (%d).\n', max_iter);
end

%% Summary
if verbose
    fprintf('\n========== Summary ==========\n');
    fprintf('Iter: %d, Obj: %.4e\n', iter, obj_now);
    fprintf('Recon: %.4e (%.1f%%), Cons: %.4e (%.1f%%), QA: %.4e (%.1f%%)\n', ...
            obj_rec, 100*obj_rec/obj_now, lambda*obj_cons, 100*lambda*obj_cons/obj_now, ...
            beta*obj_qa, 100*beta*obj_qa/obj_now);
    fprintf('||Ps^T Ps - I|| = %.2e\n', norm(Ps'*Ps - eye(c), 'fro'));
    for v = 1:V
        fprintf('  Q{%d}: min=%.3e, max=%.3e, nnz=%.1f%%\n', ...
                v, min(Q{v}(:)), max(Q{v}(:)), 100*sum(Q{v}(:)>epsilon_base)/numel(Q{v}));
    end
end

diag.final_iter = iter;
diag.final_rel  = rel_change;
diag.converged  = (iter < max_iter) && (iter > 1 && rel_change < tol);
diag.obj_terms  = [obj_rec, lambda*obj_cons, beta*obj_qa];
end

%% ==================== Helpers ====================

function [obj, obj_rec, obj_cons, obj_qa] = calc_obj(X, R, Ps, Q, A, S, lambda, beta)
obj_rec = 0; obj_cons = 0; obj_qa = 0;
for v = 1:length(X)
    Mv = Ps + Q{v};
    obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * A{v}, 'fro')^2;
    obj_cons = obj_cons + norm(Mv * A{v} - S, 'fro')^2;
    obj_qa   = obj_qa   + norm(Q{v} * A{v}, 'fro')^2;
end
obj = obj_rec + lambda * obj_cons + beta * obj_qa;
end

function S_out = col_simplex_project(S_in)
S_out = project_simplex(S_in')';
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
