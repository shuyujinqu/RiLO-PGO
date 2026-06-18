function result = posegraphSLAM_gd(varargin)
%POSEGRAPHSLAM_GD  PieADMM baseline.
% Classic penalty-based ADMM for SE(3) pose graph:
%   minimise  sum_e  || R_j - R_i*dR_e ||_F^2 + w_t*|| t_j - t_i - R_i*dt_e ||^2
% via alternating:
%   R-step : closed-form rotation averaging (SVD projection)
%   t-step : sparse linear solve for translations
%   penalty rho grows each iteration (standard ADMM schedule)
%
% This compact floor-solver implementation is included only to make the
% public RiLO-PGO reference package runnable without external solver code.

    % ---- parse args (same signatures as before) ----
    if nargin>=3 && isnumeric(varargin{1}) && isstruct(varargin{nargin})
        vertex = varargin{1}; edge = varargin{2};
        opts   = varargin{nargin};
    elseif nargin>=2 && (ischar(varargin{1})||isstring(varargin{1}))
        [vertex,edge] = parse_g2o_se3quat(char(varargin{1}));
        opts = varargin{2};
    else
        error('Bad args');
    end

    if ~isfield(opts,'MaxIter'), opts.MaxIter=200; end
    if ~isfield(opts,'beta'),    opts.beta=10;     end
    if ~isfield(opts,'rho'),     opts.rho=1.05;    end
    if ~isfield(opts,'max_beta'),opts.max_beta=500;end
    if ~isfield(opts,'tol'),     opts.tol=1e-6;    end
    if ~isfield(opts,'w_t'),     opts.w_t=1.0;     end

    num_v = size(vertex,1);
    num_e = size(edge,1);

    % Initialise from opts.p0/t0 if provided, else from vertex
    if isfield(opts, 't0') && isfield(opts, 'p0') && ~isempty(opts.t0) && ~isempty(opts.p0)
        t0_in = opts.t0;
        p0_in = opts.p0;
        if size(t0_in,1) ~= 3, t0_in = t0_in'; end
        if size(p0_in,1) ~= 4, p0_in = p0_in'; end
        t = t0_in;        % 3 x N
        R = zeros(3,3,num_v);
        for i=1:num_v
            q = p0_in(:,i); q=q/(norm(q)+1e-12);
            R(:,:,i) = quat2R(q);
        end
    else
        t = vertex(:,2:4)';        % 3 x N
        R = zeros(3,3,num_v);
        for i=1:num_v
            q = vertex(i,5:8)'; q=q/(norm(q)+1e-12);
            R(:,:,i) = quat2R(q);
        end
    end

    % Pre-compute measurement rotations and translations
    dR = zeros(3,3,num_e);
    dt = zeros(3,num_e);
    for e=1:num_e
        q_m = edge(e,6:9)'; q_m=q_m/(norm(q_m)+1e-12);
        dR(:,:,e) = quat2R(q_m);
        dt(:,e)   = edge(e,3:5)';
    end

    beta = opts.beta;

    % Build Laplacian-like sparse structure for translation solve (constant)
    % L * t_vec = rhs  (anchor first node)
    % We rebuild rhs each iter but reuse sparsity pattern.

    prev_cost = inf;
    for k=1:opts.MaxIter

        % ---- R-step: for each node, project onto SO(3) ----
        % Accumulate weighted sum of neighbours' relative rotation targets
        S = zeros(3,3,num_v);   % S(:,:,i) += beta * R_j * dR_e'  (or R_i = proj(S))
        for e=1:num_e
            i=edge(e,1); j=edge(e,2);
            % measurement: R_j ≈ R_i * dR_e  =>  R_i ≈ R_j * dR_e'
            S(:,:,i) = S(:,:,i) + beta * R(:,:,j) * dR(:,:,e)';
            S(:,:,j) = S(:,:,j) + beta * R(:,:,i) * dR(:,:,e);
        end
        for i=2:num_v   % anchor i=1
            [U,~,V] = svd(S(:,:,i));
            d = sign(det(U*V'));
            R(:,:,i) = U * diag([1,1,d]) * V';
        end

        % ---- t-step: sparse linear solve ----
        % For each edge e=(i,j): residual = t_j - t_i - R_i*dt_e
        % gradient => standard linear system
        % anchor: t_1 = 0
        % dim = 3*(num_v-1)
        dim_t = 3*(num_v-1);
        row_A=[]; col_A=[]; val_A=[];
        rhs = zeros(dim_t,1);

        for e=1:num_e
            i=edge(e,1); j=edge(e,2);
            meas = R(:,:,i)*dt(:,e);    % predicted relative translation
            res  = -meas;               % rhs contribution

            % d/d t_j ( ||t_j - t_i - meas||^2 ) -> +2*(t_j - t_i - meas)
            if j>1
                jj=3*(j-2)+(1:3);
                for r=1:3
                    row_A(end+1)=jj(r); col_A(end+1)=jj(r); val_A(end+1)=2*beta;
                end
                rhs(jj) = rhs(jj) + 2*beta*res;
            end
            if i>1
                ii=3*(i-2)+(1:3);
                for r=1:3
                    row_A(end+1)=ii(r); col_A(end+1)=ii(r); val_A(end+1)=2*beta;
                end
                rhs(ii) = rhs(ii) - 2*beta*res;
            end
            if i>1 && j>1
                ii=3*(i-2)+(1:3); jj=3*(j-2)+(1:3);
                for r=1:3
                    row_A(end+1)=ii(r); col_A(end+1)=jj(r); val_A(end+1)=-2*beta;
                    row_A(end+1)=jj(r); col_A(end+1)=ii(r); val_A(end+1)=-2*beta;
                end
            end
        end

        A_mat = sparse(row_A,col_A,val_A,dim_t,dim_t);
        A_mat = A_mat + 1e-8*speye(dim_t);   % regularise
        dt_vec = -(A_mat \ rhs);
        if any(~isfinite(dt_vec)), break; end
        for i=2:num_v
            t(:,i) = dt_vec(3*(i-2)+(1:3));
        end
        t(:,1) = zeros(3,1);

        % ---- cost ----
        cost = 0;
        for e=1:num_e
            i=edge(e,1); j=edge(e,2);
            dR_e = R(:,:,i)'*R(:,:,j) - dR(:,:,e);
            cost = cost + norm(dR_e,'fro')^2;
            err_t = t(:,j)-t(:,i)-R(:,:,i)*dt(:,e);
            cost = cost + opts.w_t * norm(err_t)^2;
        end

        if abs(prev_cost-cost)/(prev_cost+1e-12) < opts.tol, break; end
        prev_cost = cost;
        beta = min(beta*opts.rho, opts.max_beta);
    end

    % Convert back to pose7n [x y z qw qx qy qz]
    pose7n = zeros(num_v,7);
    for i=1:num_v
        pose7n(i,1:3) = t(:,i)';
        pose7n(i,4:7) = R2quat(R(:,:,i))';
    end
    result.pose7n_new = pose7n;
end

function R=quat2R(q)
    q=q(:)/(norm(q)+1e-12); w=q(1);x=q(2);y=q(3);z=q(4);
    R=[1-2*(y^2+z^2),2*(x*y-z*w),2*(x*z+y*w);
       2*(x*y+z*w),1-2*(x^2+z^2),2*(y*z-x*w);
       2*(x*z-y*w),2*(y*z+x*w),1-2*(x^2+y^2)];
end
function q=R2quat(R)
    tr=trace(R);
    if tr>0
        s=sqrt(tr+1)*2; q=[s/4;(R(3,2)-R(2,3))/s;(R(1,3)-R(3,1))/s;(R(2,1)-R(1,2))/s];
    elseif R(1,1)>R(2,2)&&R(1,1)>R(3,3)
        s=sqrt(1+R(1,1)-R(2,2)-R(3,3))*2;
        q=[(R(3,2)-R(2,3))/s;s/4;(R(1,2)+R(2,1))/s;(R(1,3)+R(3,1))/s];
    elseif R(2,2)>R(3,3)
        s=sqrt(1+R(2,2)-R(1,1)-R(3,3))*2;
        q=[(R(1,3)-R(3,1))/s;(R(1,2)+R(2,1))/s;s/4;(R(2,3)+R(3,2))/s];
    else
        s=sqrt(1+R(3,3)-R(1,1)-R(2,2))*2;
        q=[(R(2,1)-R(1,2))/s;(R(1,3)+R(3,1))/s;(R(2,3)+R(3,2))/s;s/4];
    end
    q=q/(norm(q)+1e-12);
end
