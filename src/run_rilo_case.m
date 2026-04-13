function result = run_rilo_case(g2o_file, out_dir)
%RUN_RILO_CASE Run RiLO-PGO on one SE(3) pose-graph file.

assert(exist(g2o_file, 'file') == 2, 'Cannot find g2o file: %s', g2o_file);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

[vertex, edge, ids] = parse_g2o_se3quat(g2o_file);
[p0, t0] = init_from_odometry(vertex, edge);
pose_init = [t0.' p0.'];

opts = default_rilo_opts(vertex, p0, t0);

t_start = tic;
solver_out = posegraphSLAM_Ours(vertex, edge, pose_init, opts);
cpu_time = toc(t_start);

pose7n = solver_out.pose7n_new;
[theta, t_loss] = metric_edge_sum(edge, pose7n);
[~, ~, fml] = metric_fml(edge, pose7n, 1, 1);

optimized_vertex = vertex;
optimized_vertex(:, 2:4) = pose7n(:, 1:3);
optimized_vertex(:, 5:8) = pose7n(:, 4:7);

[~, name_only, ~] = fileparts(g2o_file);
write_g2o_se3quat(fullfile(out_dir, 'optimized.g2o'), optimized_vertex, edge, ids);
write_metrics_txt(fullfile(out_dir, 'metrics.txt'), name_only, theta, t_loss, fml, cpu_time);
save(fullfile(out_dir, 'result.mat'), 'solver_out', 'pose7n', 'theta', 't_loss', 'fml', 'cpu_time', '-v7.3');
save_trajectory_plot(fullfile(out_dir, 'traj3d.png'), vertex(:, 2:4), pose7n(:, 1:3), name_only);

result = struct();
result.dataset_name = name_only;
result.theta = theta;
result.t_loss = t_loss;
result.fml = fml;
result.cpu_time = cpu_time;
result.out_dir = out_dir;
end

function opts = default_rilo_opts(vertex, p0, t0)
num_v = size(vertex, 1);

opts = struct();
opts.p0 = p0;
opts.q0 = p0;
opts.t0 = t0;
opts.lambda0 = zeros(4, num_v);

opts.MaxIter = 160;
opts.tol = 1e-6;
opts.beta = 8;
opts.rho = 1.05;
opts.max_beta = 250;
opts.H1 = 5;
opts.H2 = 5;
opts.H3 = 5;

opts.omega_t = 1.0;
opts.omega_R = 1.0;
opts.use_sl = false;
end

function write_metrics_txt(txt_file, dataset_name, theta, t_loss, fml, cpu_time)
fid = fopen(txt_file, 'w');
assert(fid >= 0, 'Cannot open %s for writing.', txt_file);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'dataset: %s\n', dataset_name);
fprintf(fid, 'theta: %.12g\n', theta);
fprintf(fid, 't_loss: %.12g\n', t_loss);
fprintf(fid, 'fML: %.12g\n', fml);
fprintf(fid, 'cpu_time_sec: %.12g\n', cpu_time);
end

function save_trajectory_plot(out_png, raw_xyz, opt_xyz, dataset_name)
fig = figure('Color', 'w', 'Position', [100 100 920 420]);

subplot(1, 2, 1);
plot3(raw_xyz(:,1), raw_xyz(:,2), raw_xyz(:,3), '-', 'Color', [0.35 0.45 0.85], 'LineWidth', 1.4);
grid on; axis equal;
title('Input Graph');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1, 2, 2);
plot3(opt_xyz(:,1), opt_xyz(:,2), opt_xyz(:,3), '-', 'Color', [0.10 0.60 0.18], 'LineWidth', 1.4);
grid on; axis equal;
title('RiLO-PGO Result');
xlabel('x'); ylabel('y'); zlabel('z');

sgtitle(sprintf('RiLO-PGO: %s', strrep(dataset_name, '_', '\_')));
saveas(fig, out_png);
close(fig);
end
