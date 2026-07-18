function [R, Ps, Q, Z, S, C, obj_history] = bcd_mvrl19(X, m, c, lambda, mu, gamma, opts)
% BCD_MVRL19  Multi-View RL with doubly-stochastic alignment C.
%
%   [R, Ps, Q, Z, S, C, obj] = bcd_mvrl19(X, m, c, lambda, mu, gamma, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})Z^{(v)}||_F^2
%       + λ Σ_v ||C^{(v)} Ps Z^{(v)} - S||_F^2     ← aligned consensus
%       + μ Σ_v Tr(D^{(v)T} C^{(v)})               ← transport cost
%       + γ Σ_v ||Z^{(v)}||_F^2                    ← ridge
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (Stiefel GD + SVD)
%     Q{v}: m×c,    unconstrained             (PGD)
%     Z{v}: c×n,    unconstrained             (PGD + simplex opt)
%     S   : m×n,    unconstrained             (mean update)
%     C{v}: m×m,    doubly stochastic          (Sinkhorn)
%
%   Update order: R → Q → Z → Ps → C
%   Data: X{v} is d_v × n

%% Options
if nargin < 7, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
eta_c     = get_opt(opts, 'eta_c',     1e-3);
eta_Ps    = get_opt(opts, 'eta_Ps',    1e-3);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL19 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f, μ=%.4f, γ=%.4f\n', n, m, c, V, lambda, mu, gamma);
    fprintf('C: doubly stochastic (Sinkhorn), Ps: Stiefel, R: Procrustes\n');
    fprintf('Update: R → Q → Z → Ps → C\n');
end
for v = 1:V, assert(size(X{v},2)==n, 'Sample mismatch.'); end

%% Init
rng(42, 'twister');
R = cell(1,V);
for v=1:V
    dv=size(X{v},1);
    if dv>=m,[R{v},~]=qr(randn(dv,m),0);else R{v}=randn(dv,m)/sqrt(dv);end
end
[Ps,~]=qr(randn(m,c),0);
Q=cell(1,V);for v=1:V,Q{v}=randn(m,c)*0.01;end
Z=cell(1,V);for v=1:V,Z{v}=randn(c,n)*0.01;end
S=zeros(m,n);  % initialized from mean
C=cell(1,V);for v=1:V,C{v}=eye(m)/m + rand(m)*0.01; C{v}=sinkhorn(max(C{v},0));end

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Orthogonal Procrustes)
    for v=1:V
        Bv=Ps+Q{v};Mv=Bv*Z{v};
        [U_R,~,V_R]=svd(X{v}*Mv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 2: Q (PGD)
    for v=1:V
        L_Q=2*norm(R{v}'*R{v},2)*norm(Z{v}*Z{v}',2);eta_Q=1/max(L_Q,eps_reg);
        for pgd=1:pgd_steps
            Bv=Ps+Q{v};
            grad_Q=2*R{v}'*(R{v}*Bv*Z{v}-X{v})*Z{v}';
            Q{v}=Q{v}-eta_Q*grad_Q;
        end
    end

    %% Step 3: Z (PGD)
    for v=1:V
        Bv=Ps+Q{v};
        L_Z=2*(norm(Bv'*R{v}'*R{v}*Bv,2)+lambda*norm(Ps'*C{v}'*C{v}*Ps,2)+gamma);
        eta_Z=1/max(L_Z,eps_reg);
        for pgd=1:pgd_steps
            grad_Z=2*Bv'*R{v}'*(R{v}*Bv*Z{v}-X{v})...
                  +2*lambda*Ps'*C{v}'*(C{v}*Ps*Z{v}-S)+2*gamma*Z{v};
            Z{v}=Z{v}-eta_Z*grad_Z;
        end
    end

    %% Step 4: Ps (Stiefel GD + SVD)
    %   ∇Ps = Σ [2 R^T(R B Z-X)Z^T + 2λ C^T(C Ps Z-S)Z^T]
    for pgd=1:pgd_steps
        grad_Ps=zeros(m,c);
        for v=1:V
            Bv=Ps+Q{v};
            grad_Ps=grad_Ps+2*R{v}'*(R{v}*Bv*Z{v}-X{v})*Z{v}';
            grad_Ps=grad_Ps+2*lambda*C{v}'*(C{v}*Ps*Z{v}-S)*Z{v}';
        end
        Ps=Ps-eta_Ps*grad_Ps;
        [U_P,~,V_P]=svd(Ps,'econ');Ps=U_P*V_P';
    end

    %% Step 5: C (gradient descent + Sinkhorn)
    %   D_{ij}^{(v)} = ||(Ps Z_v)_{i,:} - S_{j,:}||²  [m×m]
    %   ∇C = 2λ(C H H^T - S H^T) + D where H = Ps Z_v
    for v=1:V
        Hv=Ps*Z{v};                                    % m×n
        % Distance matrix D
        D_mat=zeros(m,m);
        for i=1:m
            for j=1:m
                D_mat(i,j)=norm(Hv(i,:)-S(j,:))^2;
            end
        end
        % Gradient
        grad_C=2*lambda*(C{v}*Hv*Hv'-S*Hv')+D_mat;
        C_tilde=C{v}-eta_c*grad_C;
        C{v}=sinkhorn(max(C_tilde,0));
    end

    %% Update S (mean of C·Ps·Z)
    S=zeros(m,n);
    for v=1:V,S=S+C{v}*Ps*Z{v};end
    S=S/V;

    %% Objective
    obj=0;
    for v=1:V
        Bv=Ps+Q{v};Hv=Ps*Z{v};
        obj=obj+norm(X{v}-R{v}*Bv*Z{v},'fro')^2;
        obj=obj+lambda*norm(C{v}*Hv-S,'fro')^2;
        % Per-view distance matrix for transport cost
        Dv=zeros(m,m);
        for i=1:m,for j=1:m,Dv(i,j)=norm(Hv(i,:)-S(j,:))^2;end;end
        obj=obj+mu*trace(Dv'*C{v});
        obj=obj+gamma*norm(Z{v},'fro')^2;
    end
    obj_history(iter)=obj;
    rel_chg=abs(obj_old-obj)/max(1,abs(obj_old));
    if verbose&&(mod(iter,5)==1||iter==1)
        fprintf('  %4d  %14.6e  %6.2e\n',iter,obj,rel_chg);
    end
    if rel_chg<tol,if verbose,fprintf('  Converged iter %d.\n',iter);end
        obj_history=obj_history(1:iter);break;end
    obj_old=obj;
end
if verbose&&iter>=max_iter,fprintf('  Max iters.\n');end
if verbose,fprintf('=== Summary: Iter=%d Obj=%.4e\n',iter,obj);end
end

%% Sinkhorn: doubly stochastic projection
function M = sinkhorn(M)
[m_m, ~]=size(M);M=max(M,1e-12);
for k=1:20
    M=M./sum(M,2);
    M=M./sum(M,1);
end
end

function v=get_opt(o,f,d)
if isfield(o,f),v=o.(f);else v=d;end
end
