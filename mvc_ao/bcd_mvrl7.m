function [R, Ps, Q, A, S, obj_history] = bcd_mvrl7(X, m, c, lambda, beta, opts)
% BCD_MVRL7  Multi-View RL with IRLS L21 Q (R^T R=I_c).
%
%   [R, Ps, Q, A, S, obj] = bcd_mvrl7(X, m, c, lambda, beta, opts)
%
%   Objective:
%     J = Σ_v ||X^{(v)} - R^{(v)}(Ps+Q^{(v)})A^{(v)}||_F^2
%       + λ Σ_v ||Ps A^{(v)} - S||_F^2              ← consensus: Ps only
%       + β Σ_v ||Q^{(v)}||_{2,1}                   ← row-sparsity on Q
%
%   Constraints:
%     R{v}: d_v×c,  (R{v})^T R{v} = I_c              (Orthogonal Procrustes)
%     Ps  : c×m,    Ps^T Ps = I_m                     (PGD + SVD)
%     Q{v}: c×m,    unconstrained                      (IRLS closed-form)
%     A{v}: m×n,    col-simplex (Duchi)                (PGD + simplex)
%     S   : c×n,    col-simplex (Duchi)                (mean + simplex)
%
%   NOTE: Q is c×m, D_w is m×m.  A·A^T is m×m.  Dimensions match now.
%
%   Update order: R → Ps → S → Q → A
%   Data: X{v} is d_v × n

%% Options
if nargin < 6, opts = struct(); end
max_iter  = get_opt(opts, 'max_iter',  100);
pgd_steps = get_opt(opts, 'pgd_steps', 5);
irls_iter = get_opt(opts, 'irls_iter', 3);
tol       = get_opt(opts, 'tol',       1e-5);
verbose   = get_opt(opts, 'verbose',   true);

V = length(X); [d1, n] = size(X{1}); eps_reg = 1e-8;

if verbose
    fprintf('\n========== BCD-MVRL7 ==========\n');
    fprintf('n=%d, c(R-cols)=%d, m(proto)=%d, V=%d\n', n, c, m, V);
    fprintf('λ=%.4f, β=%.4f\n', lambda, beta);
    fprintf('R: d_v×c(R^TR=I_c), Ps: c×m(P^TP=I_m), Q: c×m\n');
    fprintf('A: m×n(simplex), S: c×n(simplex)\n');
end
for v = 1:V, assert(size(X{v},2)==n, 'Sample mismatch.'); end

%% Init
rng(42,'twister');
R = cell(1,V);
for v=1:V
    dv=size(X{v},1);
    if dv>=c, [R{v},~]=qr(randn(dv,c),0); else R{v}=randn(dv,c)/sqrt(dv); end
end
if c>=m, [Ps,~]=qr(randn(c,m),0); else Ps=randn(c,m)/sqrt(c); end
Q=cell(1,V); for v=1:V, Q{v}=zeros(c,m); end
S=duchi_simplex_project(rand(c,n));
A=cell(1,V); for v=1:V, A{v}=duchi_simplex_project(rand(m,n)); end

obj_old=calc_obj(X,R,Ps,Q,A,S,lambda,beta);
obj_history=zeros(max_iter,1); obj_history(1)=obj_old;
if verbose, fprintf('Init:%.4e\n',obj_old);
    fprintf('\n  Iter   Obj            RelChg   Recon       Cons       L21\n');
    fprintf('  ----  --------------  ------  ----------  ---------  ---------\n');
end

%% BCD
for iter=1:max_iter
    %% Step 1: R (Procrustes)  d_v×c, R^T R=I_c
    for v=1:V
        Bv=(Ps+Q{v})*A{v};                     % c×n
        [U_R,~,V_R]=svd(X{v}*Bv','econ');      % d_v×c
        R{v}=U_R*V_R';
    end

    %% Step 2: Ps (PGD+SVD)  c×m, Ps^T Ps=I_m
    S_AA=zeros(m,m);
    for v=1:V, S_AA=S_AA+A{v}*A{v}'; end
    L_Ps=2*(1+lambda)*norm(S_AA,2); eta_Ps=1/max(L_Ps,eps_reg);
    for pgd=1:pgd_steps
        grad_Ps=zeros(c,m);
        for v=1:V
            Ev=X{v}-R{v}*Q{v}*A{v};
            grad_Ps=grad_Ps-2*R{v}'*(Ev-R{v}*Ps*A{v})*A{v}';
            grad_Ps=grad_Ps+2*lambda*(Ps*A{v}-S)*A{v}';
        end
        Ps=Ps-eta_Ps*grad_Ps;
        [U_p,~,V_p]=svd(Ps,'econ'); Ps=U_p*V_p';
    end

    %% Step 3: S (mean+simplex)  c×n
    S_mean=zeros(c,n);
    for v=1:V, S_mean=S_mean+Ps*A{v}; end      % Ps: c×m, A: m×n → c×n
    S=duchi_simplex_project(S_mean/V);

    %% Step 4: Q (IRLS)  c×m
    %   D_w(m×m) diag, A*A^T(m×m) — dimensions match!
    %   Q = R^T F A^T / (A A^T + β D_w + εI)
    for v=1:V
        Fv=X{v}-R{v}*Ps*A{v};                   % d_v×n
        for irls=1:irls_iter
            d_diag=1./(2*sqrt(sum(Q{v}.^2,2))+eps_reg); % c×1
            D_w=diag(d_diag);                    % c×c (diag, correct size... wait)
            % D_w is based on Q's rows. Q is c×m, so D_w should be c×c.
            % But A*A^T is m×m. We need same size!
            % FIX: use D_w for columns of Q via Q*D_w form
            % Actually: Q(AA^T) + β D_w Q = R^T F A^T  (D_w is c×c)
            % Q = (R^T F A^T) / (A A^T) ... no, this doesn't separate.
            % Proper IRLS: Q = (R^T F A^T) / (A A^T + β D_w) when D_w used as right weight
            % But D_w (c×c) and A A^T (m×m) still mismatch!
            %
            % The correct formulation:
            % Q A A^T + β D_w Q = RHS → this is Sylvester, not simple division.
            % For now: solve via Kronecker when c and m differ.
            %
            % Using Kronecker:
            % (A A^T ⊗ I_c + I_m ⊗ β D_w) vec(Q) = vec(RHS)
            AAT = A{v} * A{v}';                  % m × m
            RHS = R{v}' * Fv * A{v}';            % (c×d_v)(d_v×n)(n×m) = c×m
            K = kron(AAT, eye(c)) + kron(eye(m), beta * D_w); % cm × cm
            Q{v} = reshape(K \ RHS(:), [c, m]);  % c×m
        end
    end

    %% Step 5: A (PGD+simplex)  m×n
    for v=1:V
        Hv=R{v}*(Ps+Q{v});                       % d_v×m
        L_A=2*(norm(Hv'*Hv,2)+lambda*norm(Ps'*Ps,2)); eta_A=1/max(L_A,eps_reg);
        for pgd=1:pgd_steps
            grad_A=2*Hv'*(Hv*A{v}-X{v})+2*lambda*Ps'*(Ps*A{v}-S);
            A{v}=A{v}-eta_A*grad_A;
            A{v}=duchi_simplex_project(A{v});
        end
    end

    %% Obj & convergence
    [obj_new,obj_rec,obj_cons,obj_l21]=calc_obj(X,R,Ps,Q,A,S,lambda,beta);
    obj_history(iter)=obj_new;
    rel_chg=abs(obj_old-obj_new)/max(1,abs(obj_old));
    if verbose&&(mod(iter,5)==1||iter==1)
        fprintf('  %4d  %14.6e  %6.2e  %10.2e  %9.2e  %9.2e\n',...
                iter,obj_new,rel_chg,obj_rec,lambda*obj_cons,beta*obj_l21);
    end
    if rel_chg<tol, if verbose,fprintf('  Converged iter %d.\n',iter); end
        obj_history=obj_history(1:iter); break; end
    obj_old=obj_new;
end
if verbose&&iter>=max_iter, fprintf('  Max iters.\n'); end
if verbose, fprintf('=== Summary: Iter=%d Obj=%.4e Recon=%.4e Cons=%.4e L21=%.4e\n',...
        iter,obj_new,obj_rec,lambda*obj_cons,beta*obj_l21); end
end

%% Helpers
function [obj,obj_rec,obj_cons,obj_l21]=calc_obj(X,R,Ps,Q,A,S,lambda,beta)
obj_rec=0;obj_cons=0;obj_l21=0;
for v=1:length(X)
    Mv=Ps+Q{v};
    obj_rec=obj_rec+norm(X{v}-R{v}*Mv*A{v},'fro')^2;
    obj_cons=obj_cons+norm(Ps*A{v}-S,'fro')^2;
    for i=1:size(Q{v},1), obj_l21=obj_l21+norm(Q{v}(i,:)); end
end
obj=obj_rec+lambda*obj_cons+beta*obj_l21;
end

function X=duchi_simplex_project(Z)
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
