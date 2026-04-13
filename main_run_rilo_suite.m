% MAIN_RUN_RILO_SUITE  Batch-run RiLO-PGO on the disturbed benchmark suite.
%
% Outputs:
%   - results/suite_run/rilo_suite_results.mat
%   - results/suite_run/summary.csv
%   - per-dataset optimized graphs and trajectory figures

clear; clc; close all;

root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'src')));

suite_dir = fullfile(root, 'datasets', 'robust_suite_g2o');
out_dir = fullfile(root, 'results', 'suite_run');

if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

files = dir(fullfile(suite_dir, '*.g2o'));
assert(~isempty(files), 'No .g2o files found in %s', suite_dir);

all_results = repmat(struct( ...
    'dataset_name', '', ...
    'theta', nan, ...
    't_loss', nan, ...
    'fml', nan, ...
    'cpu_time', nan, ...
    'out_dir', ''), numel(files), 1);

fprintf('Running RiLO-PGO on %d datasets...\n', numel(files));

for k = 1:numel(files)
    g2o_file = fullfile(files(k).folder, files(k).name);
    case_out_dir = fullfile(out_dir, erase(files(k).name, '.g2o'));
    if exist(case_out_dir, 'dir') ~= 7
        mkdir(case_out_dir);
    end

    fprintf('\n[%d/%d] %s\n', k, numel(files), files(k).name);
    all_results(k) = run_rilo_case(g2o_file, case_out_dir);
end

save(fullfile(out_dir, 'rilo_suite_results.mat'), 'all_results', '-v7.3');
write_summary_csv(fullfile(out_dir, 'summary.csv'), all_results);

fprintf('\nBatch run complete.\n');
fprintf('Summary: %s\n', fullfile(out_dir, 'summary.csv'));

function write_summary_csv(csv_file, all_results)
fid = fopen(csv_file, 'w');
assert(fid >= 0, 'Cannot open %s for writing.', csv_file);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'dataset,theta,t_loss,fML,cpu_time_sec\n');
for k = 1:numel(all_results)
    fprintf(fid, '%s,%.12g,%.12g,%.12g,%.12g\n', ...
        all_results(k).dataset_name, ...
        all_results(k).theta, ...
        all_results(k).t_loss, ...
        all_results(k).fml, ...
        all_results(k).cpu_time);
end
end
