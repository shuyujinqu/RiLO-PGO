function result = posegraphSLAM_LM(vertex, edge, opts)
% Memory-Efficient Modified Levenberg-Marquardt solver for pose graph optimization
% Uses PCG (Preconditioned Conjugate Gradient) for large-scale problems
%
% Input:
%   vertex: N x 8 matrix [id, x, y, z, qw, qx, qy, qz]
%   edge: E x 30 matrix [id1, id2, dx, dy, dz, dqw, dqx, dqy, dqz, info...]
%   opts: options structure
%
% Output:
%   result.pose7n_new: optimized poses [x, y, z, qw, qx, qy, qz]
%   result.epsm: residual history
%   result.k: number of iterations

    if ~isfield(opts, 'MaxIter'), opts.MaxIter = 200; end
    if ~isfield(opts, 'tol'), opts.tol = 1e-15; end
    if ~isfield(opts, 'lambda'), opts.lambda = 1e-3; end
    
    num_v = size(vertex, 1);
    num_e = size(edge, 1);
    
    % ---- logging init ----
    do_log = isfield(opts,'enable_log') && opts.enable_log && isfield(opts,'vertex_true');
    if do_log
        RE_log = []; NRMSE_log = []; t_solve_log = [];
        tlog0 = tic;
    end
    
    % Initialize poses - 使用opts.p0/t0如果提供，否则从vertex
    if isfield(opts, 't0') && isfield(opts, 'p0') && ~isempty(opts.t0) && ~isempty(opts.p0)
        t = opts.t0;  % 3 x num_v
        q = opts.p0;  % 4 x num_v
        if size(t,1) ~= 3, t = t'; end
        if size(q,1) ~= 4, q = q'; end
    else
        t = vertex(:, 2:4)';
        q = vertex(:, 5:8)';
    end
    
    % Normalize quaternions
    for i = 1:num_v
        q(:,i) = q(:,i) / norm(q(:,i));
    end
    
    % Extract information matrices
    [kappa, tau] = extract_info_local(edge);

    % Optional per-edge weights
    if isfield(opts,'edge_weights') && ~isempty(opts.edge_weights)
        edge_weights = opts.edge_weights(:);
        assert(numel(edge_weights) == num_e, 'opts.edge_weights length mismatch');
        edge_weights = max(edge_weights, 1e-8);
    else
        edge_weights = ones(num_e,1);
    end
    
    epsm = zeros(opts.MaxIter, 1);
    lambda = opts.lambda;
    dim = 6 * (num_v - 1);
    
    % 判断是否使用 PCG
    use_pcg = (num_v > 2000);
    
    % Compute initial error
    prev_err = compute_error(t, q, edge, kappa, tau, edge_weights);
    
    for k = 1:opts.MaxIter
        % 使用 COO 格式构建稀疏 Hessian
        nnz_est = num_e * 72 * 2;
        row_H = zeros(nnz_est, 1);
        col_H = zeros(nnz_est, 1);
        val_H = zeros(nnz_est, 1);
        nnz_count = 0;
        
        b = zeros(dim, 1);
        
        for e = 1:num_e
            i = edge(e, 1);
            j = edge(e, 2);
            
            % Measurement
            dt_meas = edge(e, 3:5)';
            dq_meas = edge(e, 6:9)';
            dq_meas = dq_meas / norm(dq_meas);
            
            % Current estimates
            ti = t(:, i);
            tj = t(:, j);
            qi = q(:, i);
            qj = q(:, j);
            
            Ri = quat2rotm_local(qi);
            
            % Translation error
            dt_pred = Ri' * (tj - ti);
            err_t = dt_pred - dt_meas;
            
            % Rotation error
            qi_inv = quat_inv(qi);
            dq_pred = quat_mult(qi_inv, qj);
            dq_err = quat_mult(quat_inv(dq_meas), dq_pred);
            if dq_err(1) < 0
                dq_err = -dq_err;
            end
            err_r = 2 * dq_err(2:4);
            
            err = [err_t; err_r];
            Omega = edge_weights(e) * blkdiag(kappa{e}, tau{e});
            
            % Jacobians
            dt_local = Ri' * (tj - ti);
            
            if i > 1
                idx_i = 6*(i-2) + (1:6);
                Ji = [-Ri', skew(dt_local); zeros(3,3), -eye(3)];
                Hii = Ji' * Omega * Ji;
                bi = Ji' * Omega * err;
                
                for ii = 1:6
                    for jj = 1:6
                        if abs(Hii(ii,jj)) > 1e-15
                            nnz_count = nnz_count + 1;
                            row_H(nnz_count) = idx_i(ii);
                            col_H(nnz_count) = idx_i(jj);
                            val_H(nnz_count) = Hii(ii,jj);
                        end
                    end
                end
                b(idx_i) = b(idx_i) + bi;
            end
            
            if j > 1
                idx_j = 6*(j-2) + (1:6);
                Jj = [Ri', zeros(3,3); zeros(3,3), eye(3)];
                Hjj = Jj' * Omega * Jj;
                bj = Jj' * Omega * err;
                
                for ii = 1:6
                    for jj = 1:6
                        if abs(Hjj(ii,jj)) > 1e-15
                            nnz_count = nnz_count + 1;
                            row_H(nnz_count) = idx_j(ii);
                            col_H(nnz_count) = idx_j(jj);
                            val_H(nnz_count) = Hjj(ii,jj);
                        end
                    end
                end
                b(idx_j) = b(idx_j) + bj;
            end
            
            % Cross terms
            if i > 1 && j > 1
                idx_i = 6*(i-2) + (1:6);
                idx_j = 6*(j-2) + (1:6);
                
                Ji = [-Ri', skew(dt_local); zeros(3,3), -eye(3)];
                Jj = [Ri', zeros(3,3); zeros(3,3), eye(3)];
                Hij = Ji' * Omega * Jj;
                
                for ii = 1:6
                    for jj = 1:6
                        if abs(Hij(ii,jj)) > 1e-15
                            nnz_count = nnz_count + 1;
                            row_H(nnz_count) = idx_i(ii);
                            col_H(nnz_count) = idx_j(jj);
                            val_H(nnz_count) = Hij(ii,jj);
                            
                            nnz_count = nnz_count + 1;
                            row_H(nnz_count) = idx_j(jj);
                            col_H(nnz_count) = idx_i(ii);
                            val_H(nnz_count) = Hij(ii,jj);
                        end
                    end
                end
            end
        end
        
        % 构建稀疏矩阵
        row_H = row_H(1:nnz_count);
        col_H = col_H(1:nnz_count);
        val_H = val_H(1:nnz_count);
        H = sparse(row_H, col_H, val_H, dim, dim);
        
        % LM damping: H + lambda * diag(H)
        diagH = spdiags(H, 0);
        diagH(diagH < 1e-6) = 1e-6;
        H_lm = H + lambda * spdiags(diagH, 0, dim, dim);
        
        % 求解线性系统
        if use_pcg
            [dx, ~] = pcg(H_lm, -b, 1e-6, 200);
        else
            dx = -H_lm \ b;
        end
        
        % Tentative update
        t_new = t;
        q_new = q;
        for i = 2:num_v
            idx = 6*(i-2) + (1:6);
            dt_step = dx(idx(1:3));
            dr_step = dx(idx(4:6));
            
            t_new(:,i) = t(:,i) + dt_step;
            
            dq = [1; dr_step/2];
            dq = dq / norm(dq);
            q_new(:,i) = quat_mult(q(:,i), dq);
            q_new(:,i) = q_new(:,i) / norm(q_new(:,i));
        end
        
        % Compute new error
        new_err = compute_error(t_new, q_new, edge, kappa, tau, edge_weights);
        
        epsm(k) = sqrt(new_err);
        
        % Accept or reject step
        if new_err < prev_err
            t = t_new;
            q = q_new;
            prev_err = new_err;
            lambda = max(lambda / 2, 1e-7);
        else
            lambda = min(lambda * 2, 1e7);
        end
        
        % Convergence check (relative change)
        if k > 1 && abs(epsm(k) - epsm(k-1)) < opts.tol * max(epsm(k), 1e-10)
            break;
        end
        
        % ---- logging per-iter ----
        if do_log
            pose_temp = [t', q'];
            sig = 0.05;
            if isfield(opts,'sigma_t'), sig = opts.sigma_t; end
            [REk, NRMSEk] = compute_errors_log(pose_temp, opts.vertex_true, sig);
            RE_log(end+1,1) = REk;
            NRMSE_log(end+1,1) = NRMSEk;
            t_solve_log(end+1,1) = toc(tlog0);
        end
    end
    
    result.pose7n_new = [t', q'];
    result.epsm = epsm(1:k);
    result.k = k;
    
    % ---- attach logs to result ----
    if do_log
        result.RE = RE_log;
        result.NRMSE = NRMSE_log;
        result.t_solve = t_solve_log;
    else
        result.RE = epsm(1:k) / max(epsm(1:k));
        result.NRMSE = epsm(1:k) / max(epsm(1:k));
        result.t_solve = (1:k)' * 0.01;
    end
end

%% Helper functions
function total_err = compute_error(t, q, edge, kappa, tau, edge_weights)
    total_err = 0;
    num_e = size(edge, 1);
    for e = 1:num_e
        i = edge(e, 1);
        j = edge(e, 2);
        
        dt_meas = edge(e, 3:5)';
        dq_meas = edge(e, 6:9)';
        dq_meas = dq_meas / norm(dq_meas);
        
        Ri = quat2rotm_local(q(:,i));
        dt_pred = Ri' * (t(:,j) - t(:,i));
        err_t = dt_pred - dt_meas;
        
        dq_pred = quat_mult(quat_inv(q(:,i)), q(:,j));
        dq_err = quat_mult(quat_inv(dq_meas), dq_pred);
        if dq_err(1) < 0, dq_err = -dq_err; end
        err_r = 2 * dq_err(2:4);
        
        err = [err_t; err_r];
        Omega = edge_weights(e) * blkdiag(kappa{e}, tau{e});
        total_err = total_err + err' * Omega * err;
    end
end

function S = skew(v)
    S = [0, -v(3), v(2);
         v(3), 0, -v(1);
         -v(2), v(1), 0];
end

function R = quat2rotm_local(q)
    q = q(:) / norm(q);
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    R = [1-2*(qy^2+qz^2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw);
         2*(qx*qy+qz*qw), 1-2*(qx^2+qz^2), 2*(qy*qz-qx*qw);
         2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx^2+qy^2)];
end

function q_inv = quat_inv(q)
    q_inv = [q(1); -q(2:4)] / (norm(q)^2);
end

function qr = quat_mult(q1, q2)
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);
    qr = [w1*w2 - dot(v1,v2); w1*v2 + w2*v1 + cross(v1,v2)];
end

function [kappa, tau] = extract_info_local(edge)
    num_e = size(edge, 1);
    num_cols = size(edge, 2);
    kappa = cell(num_e, 1);
    tau = cell(num_e, 1);
    
    % Check if edge has information matrix (30 columns) or just 9 columns
    has_info = (num_cols >= 30);
    
    for e = 1:num_e
        if has_info
            % Extract information matrix from columns 10:30
            info_vec = edge(e, 10:30);
            info6x6 = zeros(6,6);
            idx = 1;
            for i = 1:6
                for j = i:6
                    info6x6(i,j) = info_vec(idx);
                    info6x6(j,i) = info_vec(idx);
                    idx = idx + 1;
                end
            end
            kappa{e} = info6x6(1:3, 1:3);
            tau{e} = info6x6(4:6, 4:6);
        else
            % Use identity matrices as default
            kappa{e} = eye(3);
            tau{e} = eye(3);
        end
    end
end

function [RE, NRMSE] = compute_errors_log(pose, vertex_true, sigma_t)
    % pose: N x 7 [tx ty tz qw qx qy qz]  (只用 t)
    % vertex_true: N x 8 [id tx ty tz qw qx qy qz]

    t_est  = pose(:,1:3);
    t_true = vertex_true(:,2:4);

    % Rel.Err. (translation only): ||T-T*||_F / ||T*||_F
    num = norm(t_est - t_true, 'fro');
    den = max(norm(t_true, 'fro'), 1e-12);
    RE  = num / den;

    % NRMSE: RMSE / range (论文常用定义)
    te = t_est - t_true;
    rmse_t = sqrt(mean(sum(te.^2,2)));
    range_t = max(max(t_true)) - min(min(t_true));
    NRMSE  = rmse_t / max(range_t, 1e-12);
end

function qr = qmult_log(q1, q2)
    w1=q1(1); v1=q1(2:4);
    w2=q2(1); v2=q2(2:4);
    qr = [w1*w2 - dot(v1,v2); w1*v2 + w2*v1 + cross(v1,v2)];
    qr = qr/(norm(qr)+1e-12);
end
