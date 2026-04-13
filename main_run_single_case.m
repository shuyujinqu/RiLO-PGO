% MAIN_RUN_SINGLE_CASE  Run RiLO-PGO on one sample graph.
%
% This script is the fastest way to check that the public package is set up
% correctly. It loads one disturbed graph, runs RiLO-PGO once, and writes
% the optimized graph and a simple trajectory figure to the results folder.

clear; clc; close all;

root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'src')));

g2o_file = fullfile(root, 'datasets', 'robust_suite_g2o', 'cubicle__loop_outlier_05.g2o');
out_dir = fullfile(root, 'results', 'single_case');

if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end

result = run_rilo_case(g2o_file, out_dir);

fprintf('\nFinished single-case run.\n');
fprintf('Dataset : %s\n', result.dataset_name);
fprintf('theta   : %.6e\n', result.theta);
fprintf('t_loss  : %.6e\n', result.t_loss);
fprintf('fML     : %.6e\n', result.fml);
fprintf('time    : %.2f s\n', result.cpu_time);
fprintf('Output  : %s\n', out_dir);
