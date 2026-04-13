
function [feat, names] = sl_extract_graph_features(vertex, edge)
%SL_EXTRACT_GRAPH_FEATURES Extract lightweight graph statistics for SL config.
% vertex: Nx8 [idx x y z qw qx qy qz]
% edge:   Mx9 [i j tx ty tz qw qx qy qz]
%
% Output:
%   feat:  1xD feature vector (double)
%   names: 1xD feature names

N = size(vertex,1);
M = size(edge,1);

ij = edge(:,1:2);
sep = abs(ij(:,1) - ij(:,2));
loop_ratio = sum(sep > 1) / max(M,1);

deg = accumarray([ij(:,1); ij(:,2)], 1, [N 1], @sum, 0);
avg_deg = mean(deg);
std_deg = std(double(deg));

t_norm = sqrt(sum(edge(:,3:5).^2,2));
t_mean = mean(t_norm);
t_std  = std(t_norm);
t_med  = median(t_norm);
t_p90  = percentile_local(t_norm, 90);

% rotation magnitude stats from edge quats
ang = zeros(M,1);
for k = 1:M
    q = edge(k,6:9).';
    q = q / max(norm(q),1e-12);
    qw = q(1);
    qw = max(-1,min(1,qw));
    ang(k) = 2*acos(abs(qw));
end
ang_mean = mean(ang);
ang_std  = std(ang);
ang_p90  = percentile_local(ang, 90);

density = M / max(N,1);
max_sep = max(sep) / max(N,1);

feat = double([N, M, density, avg_deg, std_deg, loop_ratio, max_sep, ...
               t_mean, t_std, t_med, t_p90, ...
               ang_mean, ang_std, ang_p90]);

names = {'N','M','E_over_V','deg_mean','deg_std','loop_ratio','max_sep_norm', ...
         't_mean','t_std','t_median','t_p90', ...
         'ang_mean','ang_std','ang_p90'};
end

function p = percentile_local(x, q)
x = sort(x(:));
n = numel(x);
if n==0, p = NaN; return; end
% linear interpolation percentile
r = 1 + (n-1) * (q/100);
lo = floor(r); hi = ceil(r);
lo = max(1,min(n,lo)); hi = max(1,min(n,hi));
if lo==hi
    p = x(lo);
else
    p = x(lo) + (r-lo) * (x(hi)-x(lo));
end
end
