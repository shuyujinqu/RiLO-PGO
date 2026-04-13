function [fR, fT, fML] = metric_fml(edge, pose7n, omega_t, omega_R)
%METRIC_FML  Evaluate the ML objective from Carlone et al., IROS 2015 (Eq. 6).
%
%   fML = sum_{(i,j)\in E} omega_t^2 * || t_j - t_i - R_i * t_ij ||^2
%                        + (omega_R^2/2) * || R_j - R_i * R_ij ||_F^2
%
% Notes
% - This is NOT a ground-truth error; it is the pose-graph consistency cost.
% - Indexing convention in this codebase:
%     pose7n is N-by-7, node ids are 1..N (MATLAB rows), edge(:,1:2) uses 1..N.
% - Edge format: [i j tx ty tz qw qx qy qz] with wxyz quaternion.

if nargin < 3 || isempty(omega_t), omega_t = 1; end
if nargin < 4 || isempty(omega_R), omega_R = 1; end

N = size(pose7n,1);
m = size(edge,1);

t = pose7n(:,1:3);
q = pose7n(:,4:7);

% Precompute rotations
R = zeros(3,3,N);
for n = 1:N
    R(:,:,n) = quat2rotm_local(q(n,:).');
end

fT = 0;
fR = 0;

for k = 1:m
    i = edge(k,1);
    j = edge(k,2);
    if i < 1 || i > N || j < 1 || j > N
        error('metric_fml: edge indices out of range (i=%d,j=%d,N=%d).', i, j, N);
    end

    t_ij = edge(k,3:5).';
    q_ij = edge(k,6:9).';
    R_ij = quat2rotm_local(q_ij);

    Ri = R(:,:,i);

    % translation residual
    et = (t(j,:).' - t(i,:).') - Ri * t_ij;
    fT = fT + (omega_t^2) * (et.'*et);

    % rotation chordal residual
    Rj = R(:,:,j);
    ER = Rj - Ri*R_ij;
    fR = fR + (omega_R^2/2) * sum(ER(:).^2); % ||.||_F^2
end

fML = fT + fR;
end

function R = quat2rotm_local(q)
% q = [qw qx qy qz]
q = q(:);
q = q / (norm(q) + 1e-12);
qw=q(1); qx=q(2); qy=q(3); qz=q(4);
R = [1-2*(qy^2+qz^2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw);
     2*(qx*qy + qz*qw), 1-2*(qx^2+qz^2), 2*(qy*qz - qx*qw);
     2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx^2+qy^2)];
end
