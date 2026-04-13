%% posegraphSLAM_RiemannianAA_GNC_vMF.m
%% AA-ADMM with GNC and von Mises-Fisher (vMF) Distribution for Rotation
%
% 关键改进：
%   1. vMF分布建模旋转噪声：用集中参数kappa替代高斯假设
%   2. 迭代式GNC：在优化过程中逐步收紧鲁棒核
%   3. 自适应外点权重：基于vMF似然和GNC动态更新
%   4. AA加速与GNC协同：先用GNC识别外点，再用AA加速
%
% vMF分布：
%   p(R | R_true, kappa) ∝ exp(kappa * tr(R_true' * R))
%   对于四元数：p(q | q_true, kappa) ∝ exp(kappa * |q' * q_true|)
%   kappa越大，分布越集中（噪声越小）
%
% 参考文献：
%   - Fisher, Lewis, Embleton (1993) "Statistical Analysis of Spherical Data"
%   - Chirikjian (2012) "Stochastic Models, Information Theory, and Lie Groups"

function result = posegraphSLAM_RiemannianAA_GNC_vMF(vertex, edge, Ad_Mat, opts)

%% 基本参数
num_v = size(vertex, 1);
num_edge = size(edge, 1);

if ~exist('opts', 'var'); opts = struct(); end

if isfield(opts, 'tol');         tol = opts.tol;        else, tol = 1e-4;   end
if isfield(opts, 'MaxIter');     MaxIter = opts.MaxIter;else, MaxIter = 200;end

% Auto-initialize p0/q0/t0 from vertex if not provided
if isfield(opts, 'p0')
    p0 = opts.p0;
else
    p0 = zeros(4, num_v);
    for i = 1:num_v
        p0(:,i) = vertex(i,5:8).';
        p0(:,i) = p0(:,i) / (norm(p0(:,i)) + 1e-12);
    end
end
if isfield(opts, 'q0'); q0 = opts.q0; else, q0 = p0; end
if isfield(opts, 't0')
    t0 = opts.t0;
else
    t0 = zeros(3, num_v);
    for i = 1:num_v
        t0(:,i) = vertex(i,2:4).';
    end
end
if isfield(opts, 'lambda0'); lambda0 = opts.lambda0; else, lambda0 = zeros(4,num_v); end

% Auto-initialize Sigma1/Sigma2 if not provided
if isfield(opts, 'Sigma1')
    Sigma1 = opts.Sigma1;
else
    Sigma1 = cell(num_edge, 1);
    for e = 1:num_edge
        Sigma1{e} = eye(4);
    end
end
if isfield(opts, 'Sigma2')
    Sigma2 = opts.Sigma2;
else
    Sigma2 = cell(num_edge, 1);
    for e = 1:num_edge
        Sigma2{e} = eye(4);
    end
end

Sigma1_orig = ensure_sigma_cell(Sigma1, num_edge);
Sigma2_orig = ensure_sigma_cell(Sigma2, num_edge);

% Lift to 4x4
for e = 1:num_edge
    Sigma1_orig{e} = lift_sigma_to4(Sigma1_orig{e});
    Sigma2_orig{e} = lift_sigma_to4(Sigma2_orig{e});
end

if isfield(opts, 'H1');          H1 = opts.H1;          else, H1 = 0;       end
if isfield(opts, 'H2');          H2 = opts.H2;          else, H2 = 0;       end
if isfield(opts, 'H3');          H3 = opts.H3;          else, H3 = 0;       end
if isfield(opts, 'vertex_true'); vertex_true = opts.vertex_true; else, vertex_true = []; end

%% ADMM参数 - 稳定设置
if isfield(opts, 'beta');        beta = opts.beta;      else, beta = 10;    end
if isfield(opts, 'rho');         rho = opts.rho;        else, rho = 1.03;   end
if isfield(opts, 'max_beta');    max_beta = opts.max_beta; else, max_beta = 300; end

%% AA参数 - 保守设置
if isfield(opts, 'aa_m');        aa_m = opts.aa_m;      else, aa_m = 3;     end
if isfield(opts, 'aa_damping');  aa_damping = opts.aa_damping; else, aa_damping = 0.3; end
if isfield(opts, 'aa_start_iter'); aa_start_iter = opts.aa_start_iter; else, aa_start_iter = 20; end
if isfield(opts, 'aa_sg_factor'); aa_sg_factor = opts.aa_sg_factor; else, aa_sg_factor = 1.05; end

%% GNC参数 - 温和设置
if isfield(opts, 'gnc_enable');  gnc_enable = opts.gnc_enable; else, gnc_enable = true; end
if isfield(opts, 'gnc_mu_init'); gnc_mu_init = opts.gnc_mu_init; else, gnc_mu_init = 1000; end
if isfield(opts, 'gnc_mu_final'); gnc_mu_final = opts.gnc_mu_final; else, gnc_mu_final = 100; end
if isfield(opts, 'gnc_factor');  gnc_factor = opts.gnc_factor; else, gnc_factor = 1.1; end
if isfield(opts, 'gnc_interval'); gnc_interval = opts.gnc_interval; else, gnc_interval = 20; end

%% vMF (von Mises-Fisher) 参数 - 固定kappa
if isfield(opts, 'vmf_enable');    vmf_enable = opts.vmf_enable;       else, vmf_enable = true;     end
if isfield(opts, 'vmf_kappa_init'); vmf_kappa_init = opts.vmf_kappa_init; else, vmf_kappa_init = 100;  end
if isfield(opts, 'vmf_kappa_min');  vmf_kappa_min = opts.vmf_kappa_min;   else, vmf_kappa_min = 100;   end
if isfield(opts, 'vmf_kappa_max');  vmf_kappa_max = opts.vmf_kappa_max;   else, vmf_kappa_max = 100;   end
if isfield(opts, 'vmf_adapt');      vmf_adapt = opts.vmf_adapt;           else, vmf_adapt = false;    end

% 为每条边初始化vMF集中参数kappa
vmf_kappa = ones(num_edge, 1) * vmf_kappa_init;

% 计算dt_scale用于GNC阈值
dt_norm = vecnorm(edge(:, 3:5), 2, 2);
dt_scale = median(dt_norm);
gnc_c = 3 * dt_scale;  % 鲁棒核阈值

%% 记录初始参数
beta0_used = beta;
gnc_mu = gnc_mu_init;

%% 初始化
epsm = zeros(MaxIter, 1);
RE = zeros(MaxIter, 1);
NRMSE = zeros(MaxIter, 1);
t_solve = zeros(MaxIter, 1);
stopc = 1;
k = 0;

p = p0;
q = q0;
t = t0;
lambda = lambda0;

% 边权重（GNC核心）
edge_weights = ones(num_edge, 1);

% AA历史
aa_history_x = {};
aa_history_g = {};

t_start = tic;

%% 构建稀疏矩阵
nz = num_edge * 6;
row_idx = zeros(nz, 1);
col_idx = zeros(nz, 1);
vals = zeros(nz, 1);
idx = 0;
for e = 1:num_edge
    i1 = edge(e, 1);
    j1 = edge(e, 2);
    for dd = 1:3
        idx = idx + 1;
        row_idx(idx) = 3*(e-1) + dd;
        col_idx(idx) = 3*(i1-1) + dd;
        vals(idx) = -1;
        
        idx = idx + 1;
        row_idx(idx) = 3*(e-1) + dd;
        col_idx(idx) = 3*(j1-1) + dd;
        vals(idx) = 1;
    end
end
Z_full = sparse(row_idx, col_idx, vals, num_edge*3, num_v*3);
Q_mat = Z_full(:, 4:num_v*3);

%% 预计算每个节点的边索引（避免邻接矩阵重复边问题）
% outgoing_edges{i} = 从节点i出发的所有边索引
% incoming_edges{i} = 到达节点i的所有边索引
outgoing_edges = cell(num_v, 1);
incoming_edges = cell(num_v, 1);
for e = 1:num_edge
    ii = edge(e, 1);
    jj = edge(e, 2);
    outgoing_edges{ii} = [outgoing_edges{ii}, e];
    incoming_edges{jj} = [incoming_edges{jj}, e];
end

%% 主循环
aa_used_count = 0;
aa_rejected_count = 0;
gnc_updates = 0;
vmf_updates = 0;

while (stopc > tol && k < MaxIter)
    k = k + 1;
    p0_iter = p;
    q0_iter = q;
    t0_iter = t;
    lambda0_iter = lambda;
    
    %% ========== GNC + vMF 权重更新 ==========
    if gnc_enable && mod(k, gnc_interval) == 0 && gnc_mu > gnc_mu_final
        % 计算每条边的残差
        edge_residuals_t = zeros(num_edge, 1);  % 平移残差
        edge_residuals_r = zeros(num_edge, 1);  % 旋转残差 (角度)
        vmf_cos_dist = zeros(num_edge, 1);      % vMF余弦距离 |q · q_meas|
        
        for e = 1:num_edge
            i = edge(e, 1);
            j = edge(e, 2);
            
            % Translation残差
            dt_meas = edge(e, 3:5)';
            Ri = quat2rotm_local(q(:, i));
            ti = t(:, i);
            tj = t(:, j);
            dt_est = Ri' * (tj - ti);
            edge_residuals_t(e) = norm(dt_est - dt_meas);
            
            % Rotation残差 (用于GNC)
            dq_meas = edge(e, 6:9)';
            if dq_meas(1) < 0, dq_meas = -dq_meas; end
            qi_inv = [q(1,i); -q(2:4,i)];
            dq_est = qmult_local(qi_inv, q(:,j));
            if dq_est(1) < 0, dq_est = -dq_est; end
            
            % vMF余弦距离: |q_est · q_meas| (用于vMF似然)
            vmf_cos_dist(e) = abs(dq_est' * dq_meas);
            
            % 旋转角度残差 (用于GNC)
            dq_diff = qmult_local([dq_meas(1); -dq_meas(2:4)], dq_est);
            edge_residuals_r(e) = 2 * acos(min(1, abs(dq_diff(1))));
        end
        
        % 综合残差 (用于GNC)
        edge_residuals = edge_residuals_t + dt_scale * edge_residuals_r;
        
        %% ========== vMF自适应kappa更新 ==========
        if vmf_enable && vmf_adapt
            for e = 1:num_edge
                % vMF对数似然: log p(q|kappa) ≈ kappa * cos_dist + log(C(kappa))
                % 其中 C(kappa) 是归一化常数
                % 
                % 自适应更新kappa:
                % 如果cos_dist接近1（估计与测量一致），增大kappa（更信任）
                % 如果cos_dist远离1（不一致），减小kappa（降低信任）
                
                cos_d = vmf_cos_dist(e);
                
                % 基于余弦距离更新kappa
                % kappa_new = kappa_old * exp(alpha * (cos_d - threshold))
                vmf_alpha = 0.5;  % 学习率
                vmf_threshold = 0.99;  % 期望的余弦距离
                
                kappa_factor = exp(vmf_alpha * (cos_d - vmf_threshold));
                vmf_kappa(e) = vmf_kappa(e) * kappa_factor;
                
                % 限制kappa范围
                vmf_kappa(e) = max(vmf_kappa_min, min(vmf_kappa_max, vmf_kappa(e)));
            end
            vmf_updates = vmf_updates + 1;
        end
        
        %% ========== GNC权重更新 (结合vMF) ==========
        c_sq = gnc_c^2;
        for e = 1:num_edge
            i = edge(e, 1);
            j = edge(e, 2);
            
            % 只对loop边做GNC
            if abs(i - j) > 1
                r_sq = edge_residuals(e)^2;
                
                % Geman-McClure weight
                w_gnc = (gnc_mu * c_sq) / (gnc_mu * c_sq + r_sq);
                
                if vmf_enable
                    % vMF似然权重: exp(kappa * (cos_dist - 1))
                    % 当cos_dist=1时，w_vmf=1；当cos_dist<1时，w_vmf<1
                    w_vmf = exp(vmf_kappa(e) * (vmf_cos_dist(e) - 1));
                    w_vmf = max(0.01, min(1, w_vmf));  % 限制范围
                    
                    % 综合权重: GNC权重 * vMF权重
                    edge_weights(e) = w_gnc * w_vmf;
                else
                    edge_weights(e) = w_gnc;
                end
            else
                edge_weights(e) = 1;  % odometry边保持权重1
            end
        end
        
        % 缩小mu（收紧鲁棒核）
        gnc_mu = max(gnc_mu / gnc_factor, gnc_mu_final);
        gnc_updates = gnc_updates + 1;
    end
    
    %% 应用权重到Sigma (结合vMF的kappa)
    Sigma1 = cell(1, num_edge);
    Sigma2 = cell(1, num_edge);
    for e = 1:num_edge
        w = edge_weights(e);
        
        % Sigma1: 平移信息矩阵 (不受vMF影响)
        Sigma1{e} = Sigma1_orig{e} * w;
        
        % Sigma2: 旋转信息矩阵 (结合vMF的kappa)
        if vmf_enable
            % vMF的kappa相当于旋转的精度/信息量
            % Sigma2 ∝ kappa * original_sigma * gnc_weight
            kappa_scale = vmf_kappa(e) / vmf_kappa_init;  % 归一化
            Sigma2{e} = Sigma2_orig{e} * w * kappa_scale;
        else
            Sigma2{e} = Sigma2_orig{e} * w;
        end
    end
    
    %% 重建t-update需要的矩阵（因为Sigma变了）
    Sigma1_diag = zeros(3*num_edge, 1);
    for e = 1:num_edge
        Se = Sigma1{e};
        Sigma1_diag(3*(e-1)+1:3*e) = diag(Se(2:4, 2:4));
    end
    Qs_Sigma1_diag = spdiags(Sigma1_diag, 0, 3*num_edge, 3*num_edge);
    n_t = num_v*3 - 3;
    QtSQ = Q_mat' * Qs_Sigma1_diag * Q_mat;
    A_mat = QtSQ + sparse(1:n_t, 1:n_t, H3/2*ones(n_t,1), n_t, n_t);
    QtS = Q_mat' * Qs_Sigma1_diag;
    
    %% 标准ADMM步：更新p (修正版：使用预计算边索引)
    p_new = p;
    for i = 2:num_v
        % 使用预计算的边索引（避免邻接矩阵重复边问题）
        out_edges = outgoing_edges{i};  % 从节点i出发的边
        in_edges = incoming_edges{i};   % 到达节点i的边
        
        p_M = zeros(4, 4);
        p_s = zeros(4, 1);
        
        % 处理从节点i出发的边 (i -> j)
        for idx = 1:length(out_edges)
            e = out_edges(idx);
            j_node = edge(e, 2);  % 目标节点
            tij = edge(e, 3:5);
            M1 = matrixM(q0_iter(:,i)) * matrixM([0, tij]) * diag([1,-1,-1,-1]);
            s1 = Vector_to_Q(t0_iter(:,j_node) - t0_iter(:,i));  % 4x1列向量
            % 累加Hessian和目标项
            p_M = p_M + M1' * Sigma1{e} * M1;
            p_s = p_s + M1' * Sigma1{e} * s1;
        end
        
        % 处理到达节点i的边 (j -> i)，即i作为目标节点
        for idx = 1:length(in_edges)
            e = in_edges(idx);
            j_node = edge(e, 1);  % 源节点
            dq_ij = edge(e, 6:9);
            M2 = matrixW(dq_ij) * matrixW(q0_iter(:,j_node)) * diag([1,-1,-1,-1]);
            s2 = [1;0;0;0];  % 4x1列向量
            % 累加Hessian和目标项
            p_M = p_M + M2' * Sigma2{e} * M2;
            p_s = p_s + M2' * Sigma2{e} * s2;
        end
        
        s3 = q0_iter(:,i) + 1/beta * lambda0_iter(:,i);
        p_M = p_M + beta/2*eye(4) + 1/2*H1*eye(4);
        p_s = p_s + beta/2*s3 + 1/2*H1*p0_iter(:,i);
        
        p_new(:,i) = p_M \ p_s;
        p_new(:,i) = p_new(:,i) / (norm(p_new(:,i)) + 1e-12);
        if p_new(1,i) < 0
            p_new(:,i) = -p_new(:,i);
        end
    end
    
    %% 标准ADMM步：更新q (使用预计算边索引) - 修复：同时考虑incoming edges
    q_new = q;
    for i = 2:num_v
        % 使用预计算的边索引
        out_edges = outgoing_edges{i};  % 从节点i出发的边 (i -> j)
        in_edges = incoming_edges{i};   % 到达节点i的边 (j -> i)
        
        q_W = zeros(4, 4);
        q_u = zeros(4, 1);
        
        % 处理从节点i出发的边 (i -> j): R_j = R_i * dR_ij
        for idx = 1:length(out_edges)
            e = out_edges(idx);
            j_node = edge(e, 2);  % 目标节点
            tij = edge(e, 3:5);
            dq_ij = edge(e, 6:9);
            
            W1 = matrixW(p_new(:,i))' * matrixW([0, tij]);
            u1 = Vector_to_Q(t0_iter(:,j_node) - t0_iter(:,i));  % 4x1列向量
            W2 = matrixW(dq_ij) * matrixM(p_new(:,j_node))';
            u2 = [1;0;0;0];  % 4x1列向量
            
            q_W = q_W + W1'*Sigma1{e}*W1 + W2'*Sigma2{e}*W2;
            q_u = q_u + W1'*Sigma1{e}*u1 + W2'*Sigma2{e}*u2;
        end
        
        % 处理到达节点i的边 (j -> i): R_i = R_j * dR_ji，即需要考虑i作为目标节点时的旋转约束
        for idx = 1:length(in_edges)
            e = in_edges(idx);
            j_node = edge(e, 1);  % 源节点
            dq_ji = edge(e, 6:9);
            
            % 对于边 j->i，约束是 q_i ≈ q_j * dq_ji
            % 即 dq_ji^{-1} * q_j^{-1} * q_i ≈ I
            % 这里我们构造对q_i的约束
            W3 = matrixM(p_new(:,j_node)) * matrixW(dq_ji);
            u3 = [1;0;0;0];
            
            q_W = q_W + W3'*Sigma2{e}*W3;
            q_u = q_u + W3'*Sigma2{e}*u3;
        end
        
        u3 = p_new(:,i) - 1/beta * lambda0_iter(:,i);
        q_W = q_W + beta/2*eye(4) + 1/2*H2*eye(4);
        q_u = q_u + beta/2*u3 + 1/2*H2*q0_iter(:,i);
        
        q_new(:,i) = q_W \ q_u;
        q_new(:,i) = q_new(:,i) / (norm(q_new(:,i)) + 1e-12);
        if q_new(1,i) < 0
            q_new(:,i) = -q_new(:,i);
        end
    end
    
    %% 标准ADMM步：更新t
    s = zeros(num_edge*3, 1);
    for e = 1:num_edge
        tij = edge(e, 3:5);
        sij = matrixW(p_new(:,edge(e,1)))' * (matrixM(q_new(:,edge(e,1))) * [0, tij]');
        s(3*e-2:3*e) = sij(2:4)';
    end
    
    b_vec = QtS * s + H3/2 * reshape(t0_iter(:,2:end), n_t, 1);
    t0_vec = reshape(t0_iter(:,2:end), n_t, 1);
    [t_v, ~] = pcg(A_mat, b_vec, 1e-8, 200, [], [], t0_vec);
    t_new = [t(:,1), reshape(t_v, 3, num_v-1)];
    
    %% Anderson Acceleration with Safeguard
    x_old = [p0_iter(:); q0_iter(:); t0_iter(:)];
    x_new = [p_new(:); q_new(:); t_new(:)];
    g_new = x_new - x_old;
    
    % 剔除gauge分量
    n_p = 4 * num_v;
    n_q = 4 * num_v;
    g_new(1:4) = 0;
    g_new(n_p + (1:4)) = 0;
    g_new(n_p + n_q + (1:3)) = 0;
    
    aa_history_x{end+1} = x_old;
    aa_history_g{end+1} = g_new;
    if length(aa_history_x) > aa_m + 1
        aa_history_x(1) = [];
        aa_history_g(1) = [];
    end
    
    use_aa = false;
    if k >= aa_start_iter && length(aa_history_g) >= 2
        m_k = length(aa_history_g) - 1;
        
        G = zeros(length(g_new), m_k);
        for j = 1:m_k
            G(:,j) = aa_history_g{j+1} - aa_history_g{j};
        end
        
        gamma = (G' * G + 1e-8*eye(m_k)) \ (G' * g_new);
        
        x_aa = x_new;
        for j = 1:m_k
            dx = (aa_history_x{j+1} - aa_history_x{j}) + (aa_history_g{j+1} - aa_history_g{j});
            x_aa = x_aa - gamma(j) * dx;
        end
        
        x_aa = (1 - aa_damping) * x_new + aa_damping * x_aa;
        
        p_aa = reshape(x_aa(1:n_p), 4, num_v);
        q_aa = reshape(x_aa(n_p+1:n_p+n_q), 4, num_v);
        t_aa = reshape(x_aa(n_p+n_q+1:end), 3, num_v);
        
        for i = 2:num_v
            p_aa(:,i) = p_aa(:,i) / (norm(p_aa(:,i)) + 1e-12);
            q_aa(:,i) = q_aa(:,i) / (norm(q_aa(:,i)) + 1e-12);
            if p_aa(1,i) < 0, p_aa(:,i) = -p_aa(:,i); end
            if q_aa(1,i) < 0, q_aa(:,i) = -q_aa(:,i); end
        end
        
        % 锁死gauge
        p_aa(:,1) = p0_iter(:,1);
        q_aa(:,1) = q0_iter(:,1);
        t_aa(:,1) = t0_iter(:,1);
        p_aa(:,1) = p_aa(:,1) / (norm(p_aa(:,1)) + 1e-12);
        if p_aa(1,1) < 0, p_aa(:,1) = -p_aa(:,1); end
        q_aa(:,1) = q_aa(:,1) / (norm(q_aa(:,1)) + 1e-12);
        if q_aa(1,1) < 0, q_aa(:,1) = -q_aa(:,1); end
        
        % Safeguard
        res_base = norm(p_new - q_new, 'fro');
        res_aa = norm(p_aa - q_aa, 'fro');
        
        if res_aa <= res_base * aa_sg_factor
            p = p_aa;
            q = q_aa;
            t = t_aa;
            use_aa = true;
            aa_used_count = aa_used_count + 1;
        else
            aa_rejected_count = aa_rejected_count + 1;
        end
    end
    
    if ~use_aa
        p = p_new;
        q = q_new;
        t = t_new;
        p(:,1) = p0_iter(:,1);
        q(:,1) = q0_iter(:,1);
        t(:,1) = t0_iter(:,1);
    end
    
    %% update lambda
    lambda = lambda0_iter - beta * (p - q);
    
    %% update beta
    beta = min(rho * beta, max_beta);
    
    %% 计算残差
    ex_q = q - q0_iter;
    ex_t = t - t0_iter;
    ex_lambda = lambda - lambda0_iter;
    eps_pri = norm(ex_lambda, 'fro') / beta;
    eps_dual = norm(ex_q, 'fro') * beta + norm(ex_t, 'fro') * beta;
    stopc = eps_pri + eps_dual;
    epsm(k) = stopc;
    
    t_solve(k) = toc(t_start);
    
    if ~isempty(vertex_true)
        t_est = t';
        t_true = vertex_true(:,2:4);
        num = norm(t_est - t_true, 'fro');
        den = max(norm(t_true, 'fro'), 1e-12);
        RE(k) = num / den;
        
        te = t_est - t_true;
        rmse_t = sqrt(mean(sum(te.^2,2)));
        range_t = max(max(t_true)) - min(min(t_true));
        NRMSE(k) = rmse_t / max(range_t, 1e-12);
    end
end

%% 输出
epsm = epsm(1:k);
t_solve = t_solve(1:k);
RE = RE(1:k);
NRMSE = NRMSE(1:k);

pose7n_new = [t', p'];

result.pose7n_new = pose7n_new;
result.k = k;
result.epsm = epsm;
result.t_solve = t_solve;
result.RE = RE;
result.NRMSE = NRMSE;

result.params_used = struct('beta0', beta0_used, 'beta_final', beta, 'rho', rho, 'max_beta', max_beta, ...
    'aa_m', aa_m, 'aa_damping', aa_damping, 'aa_start_iter', aa_start_iter, 'aa_sg_factor', aa_sg_factor);
result.aa_stats = struct('used', aa_used_count, 'rejected', aa_rejected_count);
result.gnc_stats = struct('enabled', gnc_enable, 'mu_final', gnc_mu, 'updates', gnc_updates);
result.edge_weights = edge_weights;

% vMF统计信息
result.vmf_stats = struct('enabled', vmf_enable, 'kappa_init', vmf_kappa_init, ...
    'kappa_min', min(vmf_kappa), 'kappa_max', max(vmf_kappa), 'kappa_mean', mean(vmf_kappa), ...
    'updates', vmf_updates);
result.vmf_kappa = vmf_kappa;  % 每条边的最终kappa值

% 统计外点检测
n_outliers_detected = sum(edge_weights < 0.5);
n_loop_edges = sum(abs(edge(:,1) - edge(:,2)) > 1);
result.outlier_stats = struct('detected', n_outliers_detected, 'loop_edges', n_loop_edges);

end


%% Helper functions
function qr = qmult_local(q1, q2)
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);
    qr = [w1*w2 - dot(v1, v2); w1*v2 + w2*v1 + cross(v1, v2)];
    qr = qr / (norm(qr) + 1e-12);
end

function R = quat2rotm_local(q)
    q = q / (norm(q) + 1e-12);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1-2*(y^2+z^2), 2*(x*y-w*z), 2*(x*z+w*y);
         2*(x*y+w*z), 1-2*(x^2+z^2), 2*(y*z-w*x);
         2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x^2+y^2)];
end

function Sigma_cell = ensure_sigma_cell(S, num_edge)
    if iscell(S)
        if numel(S) == num_edge
            Sigma_cell = reshape(S, 1, []);
            return;
        elseif numel(S) == 1
            Sigma_cell = repmat(S(:).', 1, num_edge);
            return;
        end
    end
    if isnumeric(S)
        [r,c] = size(S);
        if (r == c) && (r == 3 || r == 4)
            Sigma_cell = repmat({S}, 1, num_edge);
            return;
        end
    end
    Sigma_cell = repmat({eye(4)}, 1, num_edge);
end

function S4 = lift_sigma_to4(S)
    if all(size(S) == [4 4])
        S4 = S;
        return;
    end
    if all(size(S) == [3 3])
        S4 = zeros(4);
        S4(2:4, 2:4) = S;
        S4(1,1) = 0.01 * mean(diag(S));
        return;
    end
    S4 = eye(4);
end
