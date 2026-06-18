
function result = posegraphSLAM_Ours(vertex, edge, pose_init, opts)
%POSEGRAPHSLAM_OURS Compact RiLO-PGO backend reference implementation.
% Strategy:
%   1) Clean strong floor: PieADMM
%   2) Unweighted LM refinement from the floor solution
%   3) Loop-only residual scoring and residual reweighting
%   4) Weighted LM refinement (soft / hard)
%   5) Unified conservative candidate selection
%
% Design goal:
%   - keep the floor solution as a safe fallback
%   - accept refinement only when graph-level consistency improves
%   - evaluate all candidates on the same original unweighted graph metrics

if nargin < 4, opts = struct(); end
if nargin < 3, pose_init = []; end %#ok<NASGU>

N = size(vertex,1);
omega_t = 1; omega_R = 1;
if isfield(opts,'omega_t') && ~isempty(opts.omega_t), omega_t = opts.omega_t; end
if isfield(opts,'omega_R') && ~isempty(opts.omega_R), omega_R = opts.omega_R; end

cand = struct('name',{},'pose',{},'theta',{},'t',{},'fml',{},'time',{},'mean_loop_res',{},'wmed',{});

% ---------- candidate 1: Pie floor (must match standalone PieADMM as closely as possible) ----------
opts_pie = set_defaults_local(opts, N);
timer_floor = tic;
out1 = posegraphSLAM_gd(vertex, edge, opts_pie);
time_floor = toc(timer_floor);
pose1 = anchor_pose1_local(ensure_pose7n_local(out1.pose7n_new, edge));
[th1, tl1] = metric_edge_sum(edge, pose1);
[~,~,fml1] = metric_fml(edge, pose1, omega_t, omega_R);
[loop_mask, r_loop] = loop_residuals_local(edge, pose1, opts_pie.w_t, []);
mean_loop_res = mean_or_zero_local(r_loop(loop_mask));
cand(end+1) = mkcand_local('pie_floor', pose1, th1, tl1, fml1, time_floor, mean_loop_res, 1.0); %#ok<AGROW>

if nnz(loop_mask) == 0
    result = pack_result_local(cand(1));
    return;
end

% ---------- optional SL-guided tuning ----------
slcfg = struct('enabled',false,'lm_iter_bonus',0,'hard_pref',false,'soft_scale',1.0,'hard_scale',1.0);
if isfield(opts,'use_sl') && opts.use_sl
    slcfg = get_sl_tuning_local(vertex, edge, opts);
end

% ---------- loop-only weights from Pie solution ----------
w_soft = build_loop_weights_local(r_loop, loop_mask, 'soft', slcfg.soft_scale);
w_hard = build_loop_weights_local(r_loop, loop_mask, 'hard', slcfg.hard_scale);

fprintf('[soft weights] loops=%d | min=%.3e | med=%.3e | max=%.3e\n', ...
    nnz(loop_mask), min(w_soft(loop_mask)), median(w_soft(loop_mask)), max(w_soft(loop_mask)));
fprintf('[hard weights] loops=%d | min=%.3e | med=%.3e | max=%.3e\n', ...
    nnz(loop_mask), min(w_hard(loop_mask)), median(w_hard(loop_mask)), max(w_hard(loop_mask)));
fprintf('[loop residuals @ pie] theta(mean)=%.3e | terr(mean)=%.3e\n', ...
    mean_loop_theta_local(edge, pose1, loop_mask), mean_loop_t_local(edge, pose1, loop_mask));

% ---------- candidate 2: unweighted LM ----------
try
    w_unweighted = ones(size(edge,1), 1);
    opts_lm0 = build_weighted_lm_opts_local(opts, opts_pie, pose1, w_unweighted, N, slcfg, 'unweighted');
    timer_lm0 = tic;
    out0 = posegraphSLAM_LM(vertex, edge, opts_lm0);
    time_lm0 = toc(timer_lm0);
    pose0 = anchor_pose1_local(ensure_pose7n_local(out0.pose7n_new, edge));
    [th0, tl0] = metric_edge_sum(edge, pose0);
    [~,~,fml0] = metric_fml(edge, pose0, omega_t, omega_R);
    [~, r0] = loop_residuals_local(edge, pose0, opts_pie.w_t, loop_mask);
    cand(end+1) = mkcand_local('pie_unweighted_lm', pose0, th0, tl0, fml0, time_floor + time_lm0, mean_or_zero_local(r0(loop_mask)), 1.0); %#ok<AGROW>
catch ME
    warning('posegraphSLAM_Ours: pie_unweighted_lm failed: %s', ME.message);
end

% ---------- candidate 3: weighted LM soft ----------
try
    opts_lm = build_weighted_lm_opts_local(opts, opts_pie, pose1, w_soft, N, slcfg, 'soft');
    timer_soft = tic;
    out2 = posegraphSLAM_LM(vertex, edge, opts_lm);
    time_soft = toc(timer_soft);
    pose2 = anchor_pose1_local(ensure_pose7n_local(out2.pose7n_new, edge));
    [th2, tl2] = metric_edge_sum(edge, pose2);
    [~,~,fml2] = metric_fml(edge, pose2, omega_t, omega_R);
    [~, r2] = loop_residuals_local(edge, pose2, opts_pie.w_t, loop_mask);
    cand(end+1) = mkcand_local('pie_loop_reweight_soft', pose2, th2, tl2, fml2, time_floor + time_soft, mean_or_zero_local(r2(loop_mask)), median(w_soft(loop_mask))); %#ok<AGROW>
catch ME
    warning('posegraphSLAM_Ours: pie_loop_reweight_soft failed: %s', ME.message);
end

% ---------- candidate 4: weighted LM hard ----------
try
    opts_lm2 = build_weighted_lm_opts_local(opts, opts_pie, pose1, w_hard, N, slcfg, 'hard');
    timer_hard = tic;
    out3 = posegraphSLAM_LM(vertex, edge, opts_lm2);
    time_hard = toc(timer_hard);
    pose3 = anchor_pose1_local(ensure_pose7n_local(out3.pose7n_new, edge));
    [th3, tl3] = metric_edge_sum(edge, pose3);
    [~,~,fml3] = metric_fml(edge, pose3, omega_t, omega_R);
    [~, r3] = loop_residuals_local(edge, pose3, opts_pie.w_t, loop_mask);
    cand(end+1) = mkcand_local('pie_loop_reweight_hard', pose3, th3, tl3, fml3, time_floor + time_hard, mean_or_zero_local(r3(loop_mask)), median(w_hard(loop_mask))); %#ok<AGROW>
catch ME
    warning('posegraphSLAM_Ours: pie_loop_reweight_hard failed: %s', ME.message);
end

fprintf('--- candidates ---\n');
for ii = 1:numel(cand)
    fprintf('%s | theta=%.3e | t=%.3e | fML=%.3e | loop=%.3e | time=%.2f\n', ...
        cand(ii).name, cand(ii).theta, cand(ii).t, cand(ii).fml, cand(ii).mean_loop_res, cand(ii).time);
end

selection = select_conservative_local(cand, opts);
best = cand(selection.selected_index);

result = pack_result_local(best);
result.selected_candidate = best.name;
result.candidates = cand;
result.selection = selection;
end

function slcfg = get_sl_tuning_local(vertex, edge, opts)
slcfg = struct('enabled',false,'lm_iter_bonus',0,'hard_pref',false,'soft_scale',1.0,'hard_scale',1.0);
try
    if ~isfield(opts,'sl_model_path') || isempty(opts.sl_model_path) || exist(opts.sl_model_path,'file')~=2
        return;
    end
    S = load(opts.sl_model_path);
    if ~isfield(S,'model'), return; end
    [feat,~] = sl_extract_graph_features(vertex, edge);
    pcfg = sl_predict_config(S.model, feat);
    slcfg.enabled = true;
    slcfg.lm_iter_bonus = max(0, min(4, round((pcfg.stage1_iter + pcfg.stage2_iter)/20)));
    slcfg.hard_pref = pcfg.vmf_kappa0 > 5;
    slcfg.soft_scale = min(1.6, max(0.7, 0.8 + 0.1*log(max(pcfg.beta,1))));
    slcfg.hard_scale = min(1.4, max(0.5, 0.7 + 0.08*log(max(pcfg.max_beta,1))));
catch ME
    warning('posegraphSLAM_Ours: SL tuning disabled (%s)', ME.message);
end
end

function opts = set_defaults_local(opts, N)
if ~isfield(opts,'p0') || isempty(opts.p0), opts.p0 = []; end
if ~isfield(opts,'q0') || isempty(opts.q0), opts.q0 = opts.p0; end
if ~isfield(opts,'t0') || isempty(opts.t0), opts.t0 = []; end
if ~isfield(opts,'lambda0') || isempty(opts.lambda0), opts.lambda0 = zeros(4,N); end
if ~isfield(opts,'beta') || isempty(opts.beta), opts.beta = 10; end
if ~isfield(opts,'rho') || isempty(opts.rho), opts.rho = 1.05; end
if ~isfield(opts,'max_beta') || isempty(opts.max_beta), opts.max_beta = 500; end
if ~isfield(opts,'tol') || isempty(opts.tol), opts.tol = 1e-8; end
if ~isfield(opts,'w_t') || isempty(opts.w_t), opts.w_t = 1.0; end
if ~isfield(opts,'H1') || isempty(opts.H1), opts.H1 = 5; end
if ~isfield(opts,'H2') || isempty(opts.H2), opts.H2 = 5; end
if ~isfield(opts,'H3') || isempty(opts.H3), opts.H3 = 5; end
if ~isfield(opts,'MaxIter') || isempty(opts.MaxIter)
    if N < 100
        opts.MaxIter = 120;
    elseif N < 500
        opts.MaxIter = 150;
    elseif N < 2000
        opts.MaxIter = 180;
    else
        opts.MaxIter = 220;
    end
end
end

function opts_lm = build_weighted_lm_opts_local(opts, opts_pie, pose_seed, edge_w, N, slcfg, mode)
opts_lm = opts;
opts_lm = set_defaults_local(opts_lm, N);
opts_lm.p0 = pose_seed(:,4:7).';
opts_lm.q0 = pose_seed(:,4:7).';
opts_lm.t0 = pose_seed(:,1:3).';
opts_lm.edge_weights = edge_w(:);
opts_lm.MaxIter = min(12, max(6, round(0.05*opts_pie.MaxIter) + slcfg.lm_iter_bonus));
opts_lm.tol = 1e-12;
if strcmp(mode,'hard')
    opts_lm.lambda = 5e-2;
else
    opts_lm.lambda = 1e-1;
end
end

function [loop_mask, r2] = loop_residuals_local(edge, pose7n, wt, loop_mask)
if nargin < 4 || isempty(loop_mask)
    loop_mask = infer_loop_mask_local(edge);
end
num_e = size(edge,1);
r2 = zeros(num_e,1);
for e = 1:num_e
    i = edge(e,1); j = edge(e,2);
    qi = pose7n(i,4:7).';
    qj = pose7n(j,4:7).';
    ti = pose7n(i,1:3).';
    tj = pose7n(j,1:3).';
    qij = edge(e,6:9).'; qij = qij/(norm(qij)+1e-12);
    tij = edge(e,3:5).';

    Ri = quat2R_local(qi);
    Rj = quat2R_local(qj);
    Rpred = Ri' * Rj;
    Rmeas = quat2R_local(qij);
    Rerr = Rmeas' * Rpred;
    v = (trace(Rerr)-1)/2; v = max(-1,min(1,v));
    theta = acos(v);

    tpred = Ri' * (tj - ti);
    terr = tij - tpred;

    r2(e) = wt * norm(terr)^2 + theta^2;
end
end

function m = mean_loop_theta_local(edge, pose7n, loop_mask)
ths = zeros(nnz(loop_mask),1); idx=0;
for e=find(loop_mask(:)).'
    i=edge(e,1); j=edge(e,2);
    Ri = quat2R_local(pose7n(i,4:7).');
    Rj = quat2R_local(pose7n(j,4:7).');
    Rm = quat2R_local(edge(e,6:9).');
    Rerr = Rm'*(Ri'*Rj);
    v=(trace(Rerr)-1)/2; v=max(-1,min(1,v));
    idx=idx+1; ths(idx)=acos(v);
end
m = mean_or_zero_local(ths);
end

function m = mean_loop_t_local(edge, pose7n, loop_mask)
vals = zeros(nnz(loop_mask),1); idx=0;
for e=find(loop_mask(:)).'
    i=edge(e,1); j=edge(e,2);
    Ri = quat2R_local(pose7n(i,4:7).');
    ti = pose7n(i,1:3).'; tj = pose7n(j,1:3).';
    tpred = Ri'*(tj-ti);
    terr = edge(e,3:5).' - tpred;
    idx=idx+1; vals(idx)=norm(terr);
end
m = mean_or_zero_local(vals);
end

function w = build_loop_weights_local(r2, loop_mask, mode, scale)
num_e = numel(r2);
w = ones(num_e,1);
rl = r2(loop_mask);
if isempty(rl)
    return;
end
rls = sort(rl(:));
if strcmp(mode,'hard')
    c = percentile_local(rls, 55);
else
    c = percentile_local(rls, 75);
end
c = max(c*scale, 1e-6);
w(loop_mask) = c ./ (rl + c);
if strcmp(mode,'hard')
    w(loop_mask) = w(loop_mask).^2;
end
w = min(max(w, 0.03), 1.0);
end

function p = percentile_local(x, q)
if isempty(x)
    p = 1; return;
end
q = min(max(q,0),100);
idx = 1 + (numel(x)-1)*q/100;
lo = floor(idx); hi = ceil(idx);
if lo == hi
    p = x(lo);
else
    a = idx - lo;
    p = (1-a)*x(lo) + a*x(hi);
end
end

function mask = infer_loop_mask_local(edge)
i = edge(:,1); j = edge(:,2);
mask = abs(j-i) > 1;
end

function c = mkcand_local(name, pose, th, tl, fml, time, mean_loop_res, wmed)
c = struct('name',name,'pose',pose,'theta',th,'t',tl,'fml',fml,'time',time,'mean_loop_res',mean_loop_res,'wmed',wmed);
end

function selection = select_conservative_local(cand, opts)
%SELECT_CONSERVATIVE_LOCAL Unified graph-consistency candidate gate.
%
% The floor solution is always valid. Each non-floor candidate is evaluated
% on the original unweighted graph and is rejected if it degrades either the
% total graph objective or the rotational consistency beyond fixed global
% tolerances. Among admissible candidates, the least aggressive candidate is
% preferred only as a final tie-breaker.

cfg = selection_cfg_local(opts);
floor_cand = cand(1);
num_cand = numel(cand);

admissible = false(1, num_cand);
reasons = repmat({''}, 1, num_cand);
admissible(1) = true;
reasons{1} = 'floor fallback';

for ii = 2:num_cand
    fml_degrades = cand(ii).fml > floor_cand.fml * (1 + cfg.eps_f_degrade);
    theta_degrades = cand(ii).theta > floor_cand.theta * (1 + cfg.eps_theta_max);

    if fml_degrades
        reasons{ii} = 'rejected: fML degradation';
        continue;
    end
    if theta_degrades
        reasons{ii} = 'rejected: theta degradation';
        continue;
    end

    improves_fml = cand(ii).fml < floor_cand.fml * (1 - cfg.delta_f);
    improves_t = cand(ii).t < floor_cand.t * (1 - cfg.delta_t);
    theta_soft_ok = cand(ii).theta <= floor_cand.theta * (1 + cfg.eps_theta_soft);

    if improves_fml || (improves_t && theta_soft_ok)
        admissible(ii) = true;
        reasons{ii} = 'admissible';
    else
        reasons{ii} = 'rejected: insufficient improvement';
    end
end

pool = find(admissible);
non_floor_pool = pool(pool ~= 1);
if isempty(non_floor_pool)
    selected_index = 1;
else
    order_rows = zeros(numel(non_floor_pool), 4);
    for kk = 1:numel(non_floor_pool)
        ii = non_floor_pool(kk);
        order_rows(kk,:) = [cand(ii).fml, cand(ii).t, candidate_priority_local(cand(ii).name), ii];
    end
    order_rows = sortrows(order_rows, [1 2 3]);
    selected_index = order_rows(1,4);
end

selection = struct();
selection.selected_index = selected_index;
selection.selected_candidate = cand(selected_index).name;
selection.admissible = admissible;
selection.reasons = reasons;
selection.thresholds = cfg;
end

function cfg = selection_cfg_local(opts)
cfg = struct();
cfg.eps_f_degrade = get_opt_scalar_local(opts, 'eps_f_degrade', 0.01);
cfg.eps_theta_max = get_opt_scalar_local(opts, 'eps_theta_max', 0.15);
cfg.eps_theta_soft = get_opt_scalar_local(opts, 'eps_theta_soft', 0.05);
cfg.delta_f = get_opt_scalar_local(opts, 'delta_f', 0.01);
cfg.delta_t = get_opt_scalar_local(opts, 'delta_t', 0.10);
end

function value = get_opt_scalar_local(opts, name, default_value)
if isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end

function p = candidate_priority_local(name)
switch char(name)
    case 'pie_unweighted_lm'
        p = 1;
    case 'pie_loop_reweight_soft'
        p = 2;
    case 'pie_loop_reweight_hard'
        p = 3;
    otherwise
        p = 9;
end
end

function result = pack_result_local(best)
result.pose7n_new = best.pose;
result.selected_candidate = best.name;
result.theta = best.theta;
result.t_loss = best.t;
result.fml = best.fml;
result.mean_loop_res = best.mean_loop_res;
result.k = 0;
result.t_solve = best.time;
end

function pose = ensure_pose7n_local(pose_in, edge)
pose = pose_in;
N = max(max(edge(:,1:2)));
if size(pose,1) ~= N && size(pose,2) == N
    pose = pose';
end
assert(size(pose,1)==N && size(pose,2)==7, 'pose7n size mismatch');
end

function pose = anchor_pose1_local(pose)
pose = pose;
q1 = pose(1,4:7).';
t1 = pose(1,1:3).';
R1 = quat2R_local(q1);
Rinv = R1';
for i = 1:size(pose,1)
    qi = pose(i,4:7).';
    ti = pose(i,1:3).';
    Ri = quat2R_local(qi);
    Rn = Rinv * Ri;
    tn = Rinv * (ti - t1);
    pose(i,1:3) = tn.';
    pose(i,4:7) = rot2quat_local(Rn).';
end
pose(1,1:3) = [0 0 0];
pose(1,4:7) = [1 0 0 0];
end

function R = quat2R_local(q)
q = q(:) / (norm(q) + 1e-12);
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1-2*(y^2+z^2),   2*(x*y-z*w),   2*(x*z+y*w); ...
     2*(x*y+z*w),     1-2*(x^2+z^2), 2*(y*z-x*w); ...
     2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x^2+y^2)];
end

function q = rot2quat_local(R)
tr = trace(R);
if tr > 0
    s = sqrt(tr+1)*2;
    q = [s/4; (R(3,2)-R(2,3))/s; (R(1,3)-R(3,1))/s; (R(2,1)-R(1,2))/s];
elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
    s = sqrt(1+R(1,1)-R(2,2)-R(3,3))*2;
    q = [(R(3,2)-R(2,3))/s; s/4; (R(1,2)+R(2,1))/s; (R(1,3)+R(3,1))/s];
elseif R(2,2) > R(3,3)
    s = sqrt(1+R(2,2)-R(1,1)-R(3,3))*2;
    q = [(R(1,3)-R(3,1))/s; (R(1,2)+R(2,1))/s; s/4; (R(2,3)+R(3,2))/s];
else
    s = sqrt(1+R(3,3)-R(1,1)-R(2,2))*2;
    q = [(R(2,1)-R(1,2))/s; (R(1,3)+R(3,1))/s; (R(2,3)+R(3,2))/s; s/4];
end
q = q/(norm(q)+1e-12);
end

function t = get_time_local(out)
if isstruct(out)
    if isfield(out,'t_solve') && ~isempty(out.t_solve)
        ts = out.t_solve;
        if numel(ts) > 1, t = ts(end); else, t = ts; end
    elseif isfield(out,'time')
        t = out.time;
    else
        t = NaN;
    end
else
    t = NaN;
end
if ~isfinite(t), t = NaN; end
end

function m = mean_or_zero_local(x)
if isempty(x)
    m = 0;
else
    m = mean(x);
end
end
