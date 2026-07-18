function [R, Ps, Q, A, S, obj_history] = bcd_mvrl11(X, m, c, lambda, beta, opts)
% BCD_MVRL11  Multi-View Representation Learning (BCD with Duchi simplex).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl11(X, m, c, lambda, beta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2
%       + β Σ_v ||Q^{(v)} A^{(v)}||_1                      ← element-wise L1
%
%   Constraints:
%     R{v}: d_v×m,   (R{v})^T R{v} = I_m              (orthogonal projection)
%     Ps  : m×c,     Ps^T Ps = I_c                     (orthogonal prototype)
%     Q{v}: m×c,     unconstrained                      (view-specific offset)
%     A{v}: c×n,     A{v} ≥ 0, (A{v})^T 1_c = 1_n    (col-stochastic, Duchi)
%     S   : m×n,     S S^T = I_m                       (row-orthogonal, Procrustes)
%
%   Update order: R → Q → Ps → A → S
%
%   Data: X{v} is d_v × n  (features × samples)

%% Parse options
if nargin < 6, opts = struct(); end
max_iter = get_opt(opts, 'max_iter', 100);
verbose  = get_opt(opts, 'verbose',  true);

%% Dimensions
V = length(X);
[d1, n] = size(X{1});
eps_reg = 1e-8;

if verbose
    fprintf('\n========== BCD-MVRL11 ==========\n');
    fprintf('Samples: %d, m: %d, c: %d, Views: %d\n', n, m, c, V);
    fprintf('λ=%.4f, β=%.4f\n', lambda, beta);
    fprintf('R: d_v×m (R^T R=I_m), Ps: m×c (Ps^T Ps=I_c), Q: m×c\n');
    fprintf('A: c×n (Duchi simplex), S: m×n (S S^T=I_m Procrustes)\n');
    fprintf('Update: R(Procrustes) → Q(PGD+soft-threshold) → Ps(Procrustes SVD) → A(PGD+Duchi) → S(Procrustes)\n');
end

for v = 1:V
    assert(size(X{v},2) == n, 'All views must have same sample count.');
    if size(X{v},1) < m && verbose
        fprintf('  Note: View %d (d_v=%d < m=%d), R orthogonality relaxed.\n', v, size(X{v},1), m);
    end
end
if m < c && verbose
    fprintf('  Note: m=%d < c=%d, Ps orthogonality relaxed.\n', m, c);
end

%% Initialization
rng(42, 'twister');

% --- R^{(v)} (d_v × m, orthonormal columns) ---
R = cell(1, V);
for v = 1:V
    dv = size(X{v}, 1);
    if dv >= m
        [R{v}, ~] = qr(randn(dv, m), 0);
    else
        R{v} = randn(dv, m) / sqrt(dv);
    end
end

% --- Ps (m × c, orthonormal columns) ---
if m >= c
    [Ps, ~] = qr(randn(m, c), 0);
else
    Ps = randn(m, c) / sqrt(m);
end

% --- Q^{(v)} (m × c, unconstrained, small random) ---
Q = cell(1, V);
for v = 1:V
    Q{v} = randn(m, c) * 0.01;
end

% --- A^{(v)} (c × n, Duchi simplex) ---
A = cell(1, V);
for v = 1:V
    A{v} = duchi_simplex_project(rand(c, n));
end

% --- S (m × n, row-orthogonal) ---
[S_tmp,~]=qr(randn(n,m),0);S=S_tmp';  % S S^T = I_m

% Initial objective
obj_old = calc_objective(X, R, Ps, Q, A, S, lambda, beta);
obj_history = zeros(max_iter, 1);
obj_history(1) = obj_old;

if verbose
    fprintf('Init obj: %.4e\n', obj_old);
    fprintf('\n  Iter   Objective       RelChg    Recon       Cons       QA\n');
    fprintf('  ----  --------------  ------   ----------  ---------  ---------\n');
end

%% ==================== BCD Main Loop ====================
for iter = 1:max_iter
    %% Step 1: Update R^{(v)} (Orthogonal Procrustes: R^T R = I_m)
    %   [U,~,V] = svd(X_v M_v^T) → R_v = U V^T
    for v = 1:V
        Mv = (Ps + Q{v}) * A{v};                     % m × n
        [U_R, ~, V_R] = svd(X{v} * Mv', 'econ');     % d_v × m
        R{v} = U_R * V_R';                           % d_v × m, R^T R = I_m
    end

    %% Step 2: Update Q^{(v)} (PGD + element-wise soft-threshold for L1 on QA)
    %   Smooth gradient: ∇Q = 2 R^T(R(Ps+Q)A - X)A^T + 2λ((Ps+Q)A - S)A^T
    %   Q ← soft(Q - η·∇Q, η·β)  (element-wise L1 proximal)
    for v = 1:V
        L_Q = 2 * norm(R{v}'*R{v}, 2) * norm(A{v}*A{v}', 2) ...
             + 2 * lambda * norm(A{v}*A{v}', 2);
        eta_Q = 1 / max(L_Q, eps_reg);

        Mv = Ps + Q{v};
        E1 = R{v} * Mv * A{v} - X{v};
        E2 = Mv * A{v} - S;
        grad_Q = 2 * R{v}' * E1 * A{v}' + 2 * lambda * E2 * A{v}';
        Q_tilde = Q{v} - eta_Q * grad_Q;

        % Element-wise soft-threshold (L1 proximal)
        Q{v} = sign(Q_tilde) .* max(abs(Q_tilde) - eta_Q * beta, 0);
    end

    %% Step 3: Update Ps (Orthogonal Procrustes via SVD)
    %   H = Σ_v [R^T X A^T + λ S A^T - (1+λ) Q A A^T]
    %   SVD(H) → Ps = U V^T
    H_Ps = zeros(m, c);
    for v = 1:V
        H_Ps = H_Ps + R{v}' * X{v} * A{v}' + lambda * S * A{v}' ...
               - (1 + lambda) * Q{v} * A{v} * A{v}';
    end
    [U_P, ~, V_P] = svd(H_Ps, 'econ');
    Ps = U_P * V_P';                                 % Ps^T Ps = I_c

    %% Step 4: Update A^{(v)} (PGD + Duchi simplex projection)
    %   K = Ps+Q  [m × c]
    %   ∇A = -2 K^T R^T (X - R K A) + 2λ K^T (K A - S) + 2β Q^T Q A
    for v = 1:V
        Kv = Ps + Q{v};                              % m × c

        % Lipschitz step
        L_A = 2 * norm(Kv' * R{v}' * R{v} * Kv + lambda*(Kv'*Kv) + beta*(Q{v}'*Q{v}), 2);
        eta_A_l = 1 / max(L_A, eps_reg);

        % One gradient step
        E1 = X{v} - R{v} * Kv * A{v};                % d_v × n
        grad_A = -2 * Kv' * R{v}' * E1;              % c × n (recon)
        E2 = Kv * A{v} - S;                           % m × n
        grad_A = grad_A + 2 * lambda * Kv' * E2;     % c × n (consensus)
        grad_A = grad_A + 2 * beta * Q{v}' * Q{v} * A{v};  % QA reg

        A{v} = A{v} - eta_A_l * grad_A;
        A{v} = duchi_simplex_project(A{v});          % col-simplex
    end

    %% Step 5: Update S (Orthogonal Procrustes: S S^T = I_m)
    F_S = zeros(m, n);
    for v = 1:V
        F_S = F_S + (Ps + Q{v}) * A{v};              % m × n
    end
    F_S = F_S / V;
    [U_S, ~, V_S] = svd(F_S, 'econ');
    S = U_S * V_S';                                   % S S^T = I_m

    %% Objective & Convergence
    [obj_new, obj_rec, obj_cons, obj_qa] = calc_objective(X, R, Ps, Q, A, S, lambda, beta);
    obj_history(iter) = obj_new;

    rel_change = abs(obj_old - obj_new) / max(1.0, abs(obj_old));

    if verbose && (mod(iter,5)==1 || iter==1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e\n', ...
                iter, obj_new, rel_change, obj_rec, lambda*obj_cons, beta*obj_qa);
    end

    if rel_change < 1e-6
        if verbose, fprintf('  Converged at iter %d.\n', iter); end
        obj_history = obj_history(1:iter);
        break;
    end
    obj_old = obj_new;
end

if verbose && iter >= max_iter
    fprintf('  Max iters (%d).\n', max_iter);
end

%% Summary
if verbose
    fprintf('\n========== Summary ==========\n');
    fprintf('Iter: %d, Obj: %.4e\n', iter, obj_new);
    fprintf('Recon: %.4e (%.1f%%), Cons: %.4e (%.1f%%), QA: %.4e (%.1f%%)\n', ...
            obj_rec, 100*obj_rec/obj_new, lambda*obj_cons, 100*lambda*obj_cons/obj_new, ...
            beta*obj_qa, 100*beta*obj_qa/obj_new);
    fprintf('||R^T R-I||_avg=%.2e, ||Ps^T Ps-I||=%.2e\n', ...
            sqrt(err_R(R, m)/V), norm(Ps'*Ps - eye(c), 'fro'));
end
end

%% ==================== Duchi Simplex Projection ====================
function X = duchi_simplex_project(Z)
% DUCHI_SIMPLEX_PROJECT  Column-wise probability simplex projection.
%   Duchi et al. (2008): min 0.5||x-z||² s.t. x≥0, sum(x)=1
[M, N] = size(Z);
X = zeros(M, N);
for j = 1:N
    z = Z(:, j);
    u = sort(z, 'descend');
    cs = cumsum(u);
    rho_vec = (1:M)' .* u > (cs - 1);
    rho = find(rho_vec, 1, 'last');
    if isempty(rho), rho = 1; end
    theta = (cs(rho) - 1) / rho;
    X(:, j) = max(z - theta, 0);
end
end

%% ==================== Objective Calculation ====================
function [obj, obj_rec, obj_cons, obj_qa] = calc_objective(X, R, Ps, Q, A, S, lambda, beta)
obj_rec = 0; obj_cons = 0; obj_qa = 0;
for v = 1:length(X)
    Kv = Ps + Q{v};
    obj_rec  = obj_rec  + norm(X{v} - R{v} * Kv * A{v}, 'fro')^2;
    obj_cons = obj_cons + norm(Kv * A{v} - S, 'fro')^2;
    obj_qa   = obj_qa   + sum(sum(abs(Q{v} * A{v})));
end
obj = obj_rec + lambda * obj_cons + beta * obj_qa;
end

function e = err_R(R, m)
e = 0;
for v = 1:length(R)
    e = e + norm(R{v}'*R{v} - eye(m), 'fro')^2;
end
end

function val = get_opt(opts, field, default)
if isfield(opts, field), val = opts.(field); else, val = default; end
end
