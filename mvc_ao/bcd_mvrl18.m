function [R, Ps, Q, A, S, obj_history] = bcd_mvrl18(X, m, c, lambda, gamma, beta, opts)
% BCD_MVRL18  Multi-View RL (Laplacian from Ps + full consensus + Q≥0).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl18(X, m, c, lambda, gamma, beta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ Σ_v ||(Ps+Q^{(v)})A^{(v)} - S||_F^2           ← full consensus
%       + γ Σ_v ||A^{(v)}||_F^2                            ← ridge on A
%       + β Tr(S L S^T)                                    ← Laplacian from Ps
%
%   L: built from Ps rows: W_{ij}=exp(-||Ps(i,:)-Ps(j,:)||²/2σ²)
%
%   Constraints:
%     R{v}: d_v×m,  R^T R = I_m              (Procrustes)
%     Ps  : m×c,    Ps^T Ps = I_c             (OP via M)
%     Q{v}: m×c,    Q ≥ 0                     (PGD + ReLU)
%     A{v}: c×n,    col-simplex               (PGD + simplex)
%     S   : m×n,    col-simplex               (PGD + simplex)
%
%   Update order: R → A → Q → Ps → S
%   Data: X{v} is d_v × n

%% Options
if nargin < 7, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  50);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
sigma_L   = get_opt(opts, 'sigma_L',   1.0);
tol       = get_opt(opts, 'tol',       1e-4);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [~, n] = size(X{1}); eps_reg = 1e-6;

if verbose
    fprintf('\n========== BCD-MVRL18 ==========\n');
    fprintf('n=%d, m=%d, c=%d, V=%d, λ=%.4f, γ=%.4f, β=%.4f\n', n, m, c, V, lambda, gamma, beta);
    fprintf('L from Ps, S: simplex+Laplacian, Q≥0, Ps: OP\n');
    fprintf('Update: R → A → Q → Ps → S\n');
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
Q=cell(1,V);for v=1:V,Q{v}=max(rand(m,c)*0.1,0);end
A=cell(1,V);for v=1:V,A{v}=proj_simplex(rand(c,n));end
S=proj_simplex(rand(m,n));

obj_old=Inf;obj_history=zeros(max_iter,1);
if verbose,fprintf('\n  Iter   Obj            RelChg\n  ----  --------------  ------\n');end

%% BCD
for iter=1:max_iter
    %% Build L from Ps (prototype graph)
    sq_d = pdist2(Ps, Ps).^2;
    W_L = exp(-sq_d / (2 * sigma_L^2));
    W_L = (W_L + W_L') / 2;
    D_L = diag(sum(W_L, 2));
    L = D_L - W_L;  % m×m Laplacian

    %% Step 1: R (Orthogonal Procrustes)
    for v=1:V
        Mv=(Ps+Q{v})*A{v};
        [U_R,~,V_R]=svd(X{v}*Mv','econ');
        R{v}=U_R*V_R';
    end

    %% Step 2: A (PGD + simplex)
    %   ∇A = 2 B^T R^T(R B A-X) + 2λ B^T(B A-S) + 2γ A
    for v=1:V
        Bv=Ps+Q{v};
        L_A=2*(norm(Bv'*R{v}'*R{v}*Bv,2)+lambda*norm(Bv'*Bv,2)+gamma);
        eta_A=1/max(L_A,eps_reg);
        for pgd=1:pgd_steps
            grad_A=2*Bv'*R{v}'*(R{v}*Bv*A{v}-X{v})...
                  +2*lambda*Bv'*(Bv*A{v}-S)+2*gamma*A{v};
            A{v}=A{v}-eta_A*grad_A;
            A{v}=proj_simplex(A{v});
        end
    end

    %% Step 3: Q (PGD + ReLU)
    %   ∇Q = 2 R^T(R(Ps+Q)A-X)A^T + 2λ((Ps+Q)A-S)A^T
    for v=1:V
        L_Q=2*(norm(R{v}'*R{v},2)*norm(A{v}*A{v}',2)+lambda*norm(A{v}*A{v}',2));
        eta_Q=1/max(L_Q,eps_reg);
        for pgd=1:pgd_steps
            Mv=Ps+Q{v};
            grad_Q=2*R{v}'*(R{v}*Mv*A{v}-X{v})*A{v}'...
                  +2*lambda*(Mv*A{v}-S)*A{v}';
            Q{v}=max(Q{v}-eta_Q*grad_Q,0);
        end
    end

    %% Step 4: Ps (Orthogonal Procrustes via R^T left-multiply)
    %   C_v = R_v^T X_v - Q_v A_v,  D_v = S - Q_v A_v
    %   M_Ps = Σ_v (C_v + λ D_v) A_v^T  [m×c]
    %   SVD(M_Ps) → Ps = U V^T
    M_Ps=zeros(m,c);
    for v=1:V
        Cv=R{v}'*X{v}-Q{v}*A{v};
        Dv=S-Q{v}*A{v};
        M_Ps=M_Ps+(Cv+lambda*Dv)*A{v}';
    end
    [U_P,~,V_P]=svd(M_Ps,'econ');Ps=U_P*V_P';

    %% Step 5: S (PGD + simplex, Laplacian gradient)
    %   ∇S = 2λ(V·S - Σ M_v) + 2β S L
    sum_M=zeros(m,n);
    for v=1:V,sum_M=sum_M+(Ps+Q{v})*A{v};end
    L_S=2*lambda*V+2*beta*norm(L,2);eta_S=1/max(L_S,eps_reg);
    for pgd=1:pgd_steps
        grad_S=2*lambda*(V*S-sum_M)+2*beta*L*S;          % L is m×m, S is m×n
        S=S-eta_S*grad_S;
        S=proj_simplex(S);
    end

    %% Objective
    obj=0;
    for v=1:V
        Mv=Ps+Q{v};
        obj=obj+norm(X{v}-R{v}*Mv*A{v},'fro')^2;
        obj=obj+lambda*norm(Mv*A{v}-S,'fro')^2;
        obj=obj+gamma*norm(A{v},'fro')^2;
    end
    obj=obj+beta*trace(S'*L*S);
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

%% Helpers
function X=proj_simplex(Z)
[M,N]=size(Z);X=zeros(M,N);
for j=1:N
    z=Z(:,j);u=sort(z,'descend');cs=cumsum(u);
    rho=find((1:M)'.*u>(cs-1),1,'last');
    if isempty(rho),rho=1;end
    theta=(cs(rho)-1)/rho;X(:,j)=max(z-theta,0);
end
end

function v=get_opt(o,f,d)
if isfield(o,f),v=o.(f);else v=d;end
end
