 function [A, P, C, D, iter, obj, y1] = rcagl(X, Y, numanchor, lambda1, lambda2, rho, l)
 % m      : the number of anchor. the size of Z is m*n.
 % X      : cell array of multi-view data
 % l      : the latent dimension for A and P
 % lambda1: damping parameter for D update
 % lambda2: regularization for cross-view L1
 % rho    : regularization for nuclear norm ||A * L_A^(1/2)||_*
 %
 % Reference: Liu S, Liao Q, Wang S, et al.
 %   "Robust and Consistent Anchor Graph Learning for Multi-View Clustering"
 %   IEEE TKDE, 2024.
 %
 % --- Improvements over baseline (non-invasive) ---
 % (1) Ridge regularization on P (eps_P=1e-6) for numerical stability
 % (2) Column-simplex safety projection on C after co-clustering
 % (3) Absolute-change convergence criterion (alongside relative)
 % (4) Per-iteration convergence diagnostics

 if nargin < 7
     l = 64;
 end
 if nargin < 6
     rho = 0;
 end

 %% initialize
 maxIter = 50;
 IterMax = 50;
 m = numanchor;
 numclass = length(unique(Y));
 numview = length(X);
 numsample = size(Y,1);

 % --- hyper-parameters ---
 epsilon   = 1e-6;          % numerical stability
 eps_P     = 1e-6;          % ridge for P (minimal, just for stability)
 tol_rel   = 1e-4;          % relative objective change
 tol_abs   = 1e-6;          % absolute objective change

 %% --- Data normalization & anchor initialization ---
 XX = [];
 for p = 1 : numview
     X{p} = mapstd(X{p}', 0, 1);       % standardize each view
     XX = [XX; X{p}];
 end

 % Select m anchors via k-means on SVD-reduced concatenated features
 [XU, ~, ~] = svds(XX', m);
 rng(12, 'twister');
 [IDX, ~] = kmeans(XU, m, 'MaxIter', 100, 'Replicates', 10);

 % Initialize C with hard cluster assignments (one anchor = 1 per sample)
 C = zeros(m, numsample);
 for i = 1:numsample
     C(IDX(i), i) = 1;
 end

 % --- Initialize P, A, D (original-style: rand in (0,1)) ---
 for i = 1:numview
     di = size(X{i}, 1);
     D{i} = zeros(m, numsample);
     A{i} = rand(m, l);                 % original init
     P{i} = rand(l, di);                % original init
 end

 flag = 1;
 iter = 0;
 obj = [];
 eta_nuclear = 0.5;                     % fixed step for nuclear norm proximal

 %%
 while flag
     iter = iter + 1;

     %% --- Build anchor graph Laplacian L_A from current C ---
     B_co = C * C';                      % m x m anchor co-assignment affinity
     d_B = sqrt(sum(B_co, 2) + eps);
     S_norm = B_co ./ (d_B * d_B');
     L_A = eye(m) - S_norm + epsilon * eye(m);

     [V_L, D_L] = eig(L_A);
     d_L = diag(D_L);
     L_sqrt   = V_L * diag(sqrt(max(d_L, 0))) * V_L';
     L_invsqrt = V_L * diag(1 ./ sqrt(max(d_L, epsilon))) * V_L';

     %% 1. Optimize P{iv}  (least squares + tiny ridge for stability)
     parfor iv = 1:numview
         Z_iv = C + D{iv};
         K = Z_iv' * A{iv};              % n x l
         P{iv} = (K' * K + (epsilon + eps_P) * eye(l)) \ (K' * X{iv}');
     end

     %% 2. Optimize A{iv}  (least squares + nuclear norm proximal)
     parfor iv = 1:numview
         Z_iv = C + D{iv};

         % Step 2a: Least-squares for smooth reconstruction term
         part1 = (Z_iv * Z_iv' + epsilon * eye(m)) \ (Z_iv * X{iv}' * P{iv}');
         part2 = (P{iv} * P{iv}' + epsilon * eye(l));
         A_ls = part1 / part2;           % minimizer of smooth part

         % Step 2b: Nuclear norm proximal (fixed step, same as original)
         M = L_sqrt * A_ls;
         [U_M, S_M, V_M] = svd(M, 'econ');
         s = diag(S_M);
         s_t = max(s - eta_nuclear * rho, 0);
         M_t = U_M * diag(s_t) * V_M';
         A{iv} = L_invsqrt * M_t;
     end

     %% 3. Optimize C  (bipartite spectral co-clustering)
     %   C captures shared structure across views
     B = zeros(numsample, m);
     for iv = 1:numview
         W_iv = P{iv}' * A{iv}';          % d_i x m
         B = B + (X{iv} - W_iv * D{iv})' * W_iv;
     end
     B = B ./ numview;
     [P_c, ~, ~, y1] = coclustering_bipartite_fast_re(B, B, numclass, IterMax);
     C = P_c';
     % Safety: ensure column stochasticity (no-op for correct co-clustering)
     C = max(C, 0);
     col_sum = sum(C, 1);
     C = C ./ (col_sum + epsilon);

     %% 4. Optimize D{iv}  (ridge LS + cross-view L1 proximal, original formula)
     for iv = 1:numview
         sumother = 0;
         for ij = 1:numview
             if ij ~= iv
                 sumother = sumother + abs(D{ij});
             end
         end
         W_iv = P{iv}' * A{iv}';          % d_i x m

         % Step 4a: Ridge-regularized least squares
         H = (W_iv' * W_iv + lambda1 * eye(m)) \ (W_iv' * (X{iv} - W_iv * C));

         % Step 4b: L1 proximal (original step formula)
         threshold = 0.5 * lambda2 ./ (1 + lambda1) * sumother;
         D{iv} = proxF_l1(H, threshold);
     end

     %% 5. Calculate Objective Function
     term1 = 0;
     term3 = 0;
     term_rho = 0;

     for iv = 1:numview
         W_iv = P{iv}' * A{iv}';
         term1 = term1 + norm(X{iv} - W_iv * (C + D{iv}), 'fro')^2;

         temp = 0;
         for ij = 1:numview
             if ij ~= iv
                 temp = temp + sum(abs(D{ij}(:) .* D{iv}(:)));
             end
         end
         term3 = term3 + temp;

         M = L_sqrt * A{iv};
         term_rho = term_rho + sum(svd(M, 'econ'));
     end

     obj(iter) = term1 + lambda2 * term3 + rho * term_rho;

     % --- Store per-iteration breakdown for diagnostics ---
     obj_term1(iter) = term1;
     obj_term3(iter) = lambda2 * term3;
     obj_rho(iter)   = rho * term_rho;

     % --- Live display every 5 iterations ---
     if mod(iter, 5) == 1 || iter == 1
         if iter == 1
             fprintf('\n  Iter   Objective       Recon        L1(D)       Nuclear     RelChg\n');
             fprintf('  ----  --------------  ----------  ----------  ----------  ------\n');
         end
         if iter > 1
             rc = abs((obj(iter-1)-obj(iter))/max(obj(iter-1),epsilon));
         else
             rc = 0;
         end
         fprintf('  %4d  %14.4e  %10.2e  %10.2e  %10.2e  %6.2e\n', ...
                 iter, obj(iter), term1, lambda2*term3, rho*term_rho, rc);
     end

     %% 6. Convergence check
     if iter > 1
         rel_change = abs((obj(iter - 1) - obj(iter)) / max(obj(iter - 1), epsilon));
         abs_change = abs(obj(iter - 1) - obj(iter));

         if rel_change < tol_rel || abs_change < tol_abs || obj(iter) < 1e-10
             flag = 0;
         end
     end
     if iter >= maxIter
         flag = 0;
     end
 end

 %% --- Convergence diagnostic summary ---
 fprintf('\n  ===== Convergence Summary (total %d iterations) =====\n', iter);
 fprintf('  Final objective : %14.6e\n', obj(end));
 fprintf('  Recon (term1)   : %14.6e  (%5.1f%% of total)\n', ...
         obj_term1(end), 100*obj_term1(end)/obj(end));
 fprintf('  L1-D  (lambda2) : %14.6e  (%5.1f%% of total)\n', ...
         obj_term3(end), 100*obj_term3(end)/obj(end));
 fprintf('  Nuclear (rho)   : %14.6e  (%5.1f%% of total)\n', ...
         obj_rho(end), 100*obj_rho(end)/obj(end));

 if obj_term1(end) / obj(end) > 0.99
     fprintf('  >>> NOTE: Reconstruction dominates. Consider larger lambda2/rho.\n');
 end
 if iter >= maxIter
     fprintf('  >>> NOTE: Reached max iterations. May not be converged.\n');
 end
 if abs(obj(end) - obj(max(1,end-1))) / max(obj(max(1,end-1)), epsilon) < tol_rel
     fprintf('  >>> Converged (relative change < %.0e).\n', tol_rel);
 end
 fprintf('\n');
 end
