
function [theta_sum, t_sum] = metric_edge_sum(edge, pose7n)
%METRIC_EDGE_SUM Sum of rotation and translation residuals over all edges.
% edge: Mx9 [i j tx ty tz qw qx qy qz] where t is in frame i and q is Rij
% pose7n: Nx7 [x y z qw qx qy qz]

N = size(pose7n,1);
if size(pose7n,2) ~= 7, error('pose7n must be Nx7'); end
M = size(edge,1);

theta_sum = 0;
t_sum = 0;

for k = 1:M
    i = edge(k,1); j = edge(k,2);
    if i<1 || i>N || j<1 || j>N, continue; end

    ti = pose7n(i,1:3).'; tj = pose7n(j,1:3).';
    qi = pose7n(i,4:7).'; qj = pose7n(j,4:7).';
    qi = qi/max(norm(qi),1e-12); qj=qj/max(norm(qj),1e-12);
    Ri = quat2rotm_local(qi); Rj = quat2rotm_local(qj);

    t_ij = edge(k,3:5).';
    q_ij = edge(k,6:9).'; q_ij=q_ij/max(norm(q_ij),1e-12);
    R_ij = quat2rotm_local(q_ij);

    Rhat_ij = Ri.' * Rj;
    Delta = R_ij.' * Rhat_ij;
    c = (trace(Delta)-1)/2; c = max(-1,min(1,c));
    theta_sum = theta_sum + acos(c);

    et = (tj - ti) - Ri * t_ij;
    t_sum = t_sum + (et.'*et);
end
end

function R = quat2rotm_local(q)
qw=q(1); qx=q(2); qy=q(3); qz=q(4);
n = sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
if n<1e-12, qw=1;qx=0;qy=0;qz=0; else, qw=qw/n;qx=qx/n;qy=qy/n;qz=qz/n; end
R = [1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw);
     2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw);
     2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)];
end
