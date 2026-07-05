 function [A, P, C, D, iter, obj, y1] = rcagl(X, Y, numanchor, lambda1, lambda2, rho, l)
 % m      : the number of anchor. the size of Z is m*n.
 % X      : cell array of multi-view data
 % l      : the latent dimension for A and P
 % lambda1: damping parameter for D update
 % lambda2: regularization for cross-view L1
 % rho    : regularization for nuclear norm ||A * L_A^(1/2)||_*
 
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
 C = zeros(m, numsample);
 XX = [];
 
 for p = 1 : numview
     X{p} = mapstd(X{p}', 0, 1);
     XX = [XX; X{p}];
 end
 [XU, ~, ~] = svds(XX', m);
 rand('twister', 12);
 [IDX, ~] = kmeans(XU, m, 'MaxIter', 100, 'Replicates', 10);
 for i = 1:numsample
     C(IDX(i), i) = 1;
 end
 
 for i = 1:numview
     di = size(X{i}, 1);
     D{i} = zeros(m, numsample);
     A{i} = rand(m, l);
     P{i} = rand(l, di);
 end
 
 flag = 1;
 iter = 0;
 eta_nuclear = 0.5;
 epsilon = 1e-6;
 
 %%
 while flag
     iter = iter + 1;
 
     % --- Build anchor graph Laplacian L_A from current C ---
     % C (m x n), rows = anchors, cols = samples.
     % C'' * C is the m x m anchor co-assignment affinity.
     B_co = C * C';
     d_B = sqrt(sum(B_co, 2) + eps);
     S_norm = B_co ./ (d_B * d_B');
     L_A = eye(m) - S_norm + epsilon * eye(m);
 
     [V_L, D_L] = eig(L_A);
     d_L = diag(D_L);
     L_sqrt  = V_L * diag(sqrt(max(d_L, 0))) * V_L';
     L_invsqrt = V_L * diag(1 ./ sqrt(max(d_L, epsilon))) * V_L';
 
     %% 1. Optimize P{iv}
     parfor iv = 1:numview
         Z_iv = C + D{iv};
         K = Z_iv' * A{iv};
         P{iv} = (K' * K + epsilon * eye(l)) \ (K' * X{iv}');
     end
 
     %% 2. Optimize A{iv} with nuclear norm regularization
     parfor iv = 1:numview
         Z_iv = C + D{iv};
 
         % Step 2a: Least-squares for smooth reconstruction term
         part1 = (Z_iv * Z_iv' + epsilon * eye(m)) \ (Z_iv * X{iv}' * P{iv}');
         part2 = (P{iv} * P{iv}' + epsilon * eye(l));
         A_ls = part1 / part2;
 
         % Step 2b: Nuclear norm proximal step via SVD soft-thresholding
         M = L_sqrt * A_ls;
         [U_M, S_M, V_M] = svd(M, 'econ');
         s = diag(S_M);
         s_t = max(s - eta_nuclear * rho, 0);
         M_t = U_M * diag(s_t) * V_M';
         A{iv} = L_invsqrt * M_t;
     end
 
     %% 3. Optimize C
     B = zeros(numsample, m);
     for iv = 1:numview
         W_iv = P{iv}' * A{iv}';
         B = B + (X{iv} - W_iv * D{iv})' * W_iv;
     end
     B = B ./ numview;
     [P_c, ~, ~, y1] = coclustering_bipartite_fast_re(B, B, numclass, IterMax);
     C = P_c';
 
     %% 4. Optimize D{iv}
     for iv = 1:numview
         sumother = 0;
         for ij = 1:numview
             if ij ~= iv
                 sumother = sumother + abs(D{ij});
             end
         end
         W_iv = P{iv}' * A{iv}';
         H = (W_iv' * W_iv + lambda1 * eye(m)) \ (W_iv' * (X{iv} - W_iv * C));
         D{iv} = proxF_l1(H, 0.5 * lambda2 ./ (1 + lambda1) * sumother);
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
 
     if (iter > 1) && ...
        (abs((obj(iter - 1) - obj(iter)) / obj(iter - 1)) < 1e-4 || ...
         iter > maxIter || obj(iter) < 1e-10)
         flag = 0;
     end
 end
 end
 
