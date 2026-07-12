function [R, Ps, Q, Z, S, obj_history] = bcd_mvrl12(X, m, c, lambda, opts)
% BCD_MVRL12  多视图表示学习 (Procrustes R, Ps-only consensus, PGD Z).
%
%   [R, Ps, Q, Z, S, obj] = bcd_mvrl12(X, m, c, lambda, opts)
%
%   目标函数：
%     L = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})Z^{(v)}||_F^2
%       + λ Σ_v ||Ps Z^{(v)} - S||_F^2                  ← Ps-only consensus
%
%   约束：
%     R{v}: d_v×m,  (R{v})^T R{v} = I_m              (Orthogonal Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c                     (PGD + SVD)
%     Q{v}: m×c,    unconstrained                      (closed-form LS)
%     Z{v}: c×n,    col-simplex                        (PGD + simplex)
%     S   : m×n,    col-simplex                        (mean + simplex)
%       NOTE: S is m×n to match Ps·Z = (m×c)(c×n) = m×n
%
%   更新顺序：R → Ps → Q → S → Z
%   数据格式：X{v} 为 d_v × n

%% Options
if nargin < 5, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter', 100);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
alpha_P   = get_opt(opts, 'alpha_P',  1e-3);   % Ps learning rate
alpha_Z   = get_opt(opts, 'alpha_Z',  1e-3);   % Z learning rate
tol       = get_opt(opts, 'tol',      1e-4);
verbose   = get_opt(opts, 'verbose',  true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL12 ==========\n');
    fprintf('Samples: %d, m: %d, c: %d, Views: %d\n', n, m, c, V);
    fprintf('λ=%.4f, α_P=%.1e, α_Z=%.1e\n', lambda, alpha_P, alpha_Z);
    fprintf('R: d_v×m(R^TR=I_m), Ps: m×c(P^TP=I_c), Q: m×c\n');
    fprintf('Z: c×n(simplex), S: m×n(simplex)\n');
    fprintf('Update: R → Ps → Q → S → Z\n');
end
for v = 1:V, assert(size(X{v},2)==n, 'Sample count mismatch.'); end

%% Initialization
rng(42, 'twister');

R = cell(1, V);
for v = 1:V
    dv = size(X{v}, 1);
    if dv >= m, [R{v},~] = qr(randn(dv,m),0); else, R{v}=randn(dv,m)/sqrt(dv); end
end

if m >= c, [Ps,~] = qr(randn(m,c),0); else, Ps=randn(m,c)/sqrt(m); end

Q = cell(1, V);
for v = 1:V, Q{v} = randn(m, c) * 0.01; end

Z = cell(1, V);
for v = 1:V, Z{v} = project_simplex(rand(c, n)); end

S = project_simplex(rand(m, n));

prev_loss = calc_loss(X, R, Ps, Q, Z, S, lambda);
obj_history = zeros(max_iter, 1); obj_history(1) = prev_loss;

if verbose
    fprintf('Init loss: %.4e\n', prev_loss);
    fprintf('\n  Iter   Loss           RelChg    Recon       Cons\n');
    fprintf('  ----  --------------  ------   ----------  ---------\n');
end

%% BCD Main Loop
for iter = 1:max_iter
    %% Step 1: Update R^{(v)} (Orthogonal Procrustes)
    %   B = (Ps+Q)Z  [m × n]
    %   M = X B^T → [U,~,V]=svd(M) → R=U V^T
    for v = 1:V
        Bv = (Ps + Q{v}) * Z{v};                     % m × n
        [U_R, ~, V_R] = svd(X{v} * Bv', 'econ');     % d_v × m
        R{v} = U_R * V_R';                           % R^T R = I_m
    end

    %% Step 2: Update Ps (PGD + SVD Stiefel projection)
    %   ∇Ps = Σ_v [-2 R^T (X - R(Ps+Q)Z) Z^T + 2λ (Ps Z - S) Z^T]
    for pgd = 1:pgd_steps
        grad_Ps = zeros(m, c);
        for v = 1:V
            Mv = Ps + Q{v};
            E1 = X{v} - R{v} * Mv * Z{v};
            grad_Ps = grad_Ps - 2 * R{v}' * E1 * Z{v}';
            E2 = Ps * Z{v} - S;
            grad_Ps = grad_Ps + 2 * lambda * E2 * Z{v}';
        end
        Ps = Ps - alpha_P * grad_Ps;
        % SVD projection to Stiefel manifold
        [U_p, ~, V_p] = svd(Ps, 'econ');
        Ps = U_p * V_p';                             % Ps^T Ps = I_c
    end

    %% Step 3: Update Q^{(v)} (closed-form least squares)
    %   F = X - R Ps Z
    %   Q = R^T F Z^T (Z Z^T + εI)^{-1}
    for v = 1:V
        Fv = X{v} - R{v} * Ps * Z{v};                % d_v × n
        Q{v} = R{v}' * Fv * Z{v}' / (Z{v} * Z{v}' + eps_reg * eye(c));  % m × c
    end

    %% Step 4: Update S (mean + simplex projection)
    %   S = simplex( (1/V) Σ_v Ps Z^{(v)} )
    S_mean = zeros(m, n);
    for v = 1:V
        S_mean = S_mean + Ps * Z{v};                 % (m×c)(c×n) = m×n
    end
    S = project_simplex(S_mean / V);

    %% Step 5: Update Z^{(v)} (PGD + simplex projection)
    %   H = R(Ps+Q)  [d_v × c]
    %   ∇Z = 2 H^T (H Z - X) + 2λ Ps^T (Ps Z - S)
    for v = 1:V
        Hv = R{v} * (Ps + Q{v});                     % d_v × c
        for pgd = 1:pgd_steps
            grad_Z = 2 * Hv' * (Hv * Z{v} - X{v}) ...
                   + 2 * lambda * Ps' * (Ps * Z{v} - S);  % c × n
            Z{v} = Z{v} - alpha_Z * grad_Z;
            Z{v} = project_simplex(Z{v});            % col-simplex
        end
    end

    %% Loss & Convergence
    [loss, obj_rec, obj_cons] = calc_loss(X, R, Ps, Q, Z, S, lambda);
    obj_history(iter) = loss;

    rel_change = abs(prev_loss - loss) / max(abs(prev_loss), eps_reg);
    if verbose && (mod(iter,5)==1 || iter==1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e\n', ...
                iter, loss, rel_change, obj_rec, lambda*obj_cons);
    end
    if rel_change < tol
        if verbose, fprintf('  Converged at iter %d.\n', iter); end
        obj_history = obj_history(1:iter);
        break;
    end
    prev_loss = loss;
end

if verbose && iter >= max_iter, fprintf('  Max iters.\n'); end
if verbose
    fprintf('=== Summary: Iter=%d Loss=%.4e Recon=%.4e Cons=%.4e\n', ...
            iter, loss, obj_rec, lambda*obj_cons);
end
end

%% ==================== Helpers ====================

function [loss, obj_rec, obj_cons] = calc_loss(X, R, Ps, Q, Z, S, lambda)
obj_rec = 0; obj_cons = 0;
for v = 1:length(X)
    Mv = Ps + Q{v};
    obj_rec  = obj_rec  + norm(X{v} - R{v} * Mv * Z{v}, 'fro')^2;
    obj_cons = obj_cons + norm(Ps * Z{v} - S, 'fro')^2;
end
loss = obj_rec + lambda * obj_cons;
end

function X = project_simplex(Z)
% 逐列概率单纯形投影 (Duchi 2008)
[M, N] = size(Z); X = zeros(M, N);
for j = 1:N
    z = Z(:, j); u = sort(z, 'descend'); cs = cumsum(u);
    rho = find((1:M)' .* u > (cs - 1), 1, 'last');
    if isempty(rho), rho = 1; end
    theta = (cs(rho) - 1) / rho;
    X(:, j) = max(z - theta, 0);
end
end

function v = get_opt(o, f, d)
if isfield(o, f), v = o.(f); else, v = d; end
end
