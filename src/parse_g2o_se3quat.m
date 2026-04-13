function [vertex, edge, ids] = parse_g2o_se3quat(g2o_file)
%PARSE_G2O_SE3QUAT Parse 3D g2o with VERTEX_SE3:QUAT and EDGE_SE3:QUAT.
% Output:
%   vertex: Nx8  [idx, x, y, z, qw, qx, qy, qz] where idx=1..N (sequential)
%   edge:   Mx30 [i, j, tx, ty, tz, qw, qx, qy, qz, info_upper_tri(21)]
%           If the information matrix is absent, the last 21 entries are the
%           identity 6x6 upper-triangular vector.
%   ids:    Nx1 original g2o vertex IDs, aligned with vertex rows.

fid = fopen(g2o_file, 'r');
if fid < 0
    error('Cannot open g2o file: %s', g2o_file);
end

v_ids  = zeros(0,1);
v_xyzq = zeros(0,7);   % x y z qx qy qz qw (g2o order)
e_ij   = zeros(0,2);
e_meas = zeros(0,7);   % tx ty tz qx qy qz qw

def_info = upper_tri_from_info6(eye(6));
e_info = zeros(0,21);

while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strtrim(line);
    if isempty(line) || startsWith(line,'#')
        continue;
    end

    if startsWith(line, 'VERTEX_SE3:QUAT')
        a = sscanf(line, 'VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f');
        if numel(a) ~= 8, continue; end
        v_ids(end+1,1) = a(1);
        v_xyzq(end+1,:) = [a(2) a(3) a(4) a(5) a(6) a(7) a(8)];

    elseif startsWith(line, 'EDGE_SE3:QUAT')
        toks = strsplit(line);
        if numel(toks) < 10, continue; end
        nums = str2double(toks(2:end));
        if any(isnan(nums(1:9))), continue; end

        e_ij(end+1,:)   = nums(1:2);
        e_meas(end+1,:) = nums(3:9);

        info = def_info;
        if numel(nums) >= 30
            info = nums(10:30);
        elseif numel(nums) > 9
            k = min(21, numel(nums)-9);
            info(1:k) = nums(10:9+k);
        end
        e_info(end+1,:) = info;
    end
end
fclose(fid);

% sort vertices by original ID, build id->row map
[ids, perm] = sort(v_ids);
v_xyzq = v_xyzq(perm,:);
N = numel(ids);

id2row = containers.Map('KeyType','int64','ValueType','int64');
for k = 1:N
    id2row(int64(ids(k))) = int64(k);
end

% build vertex (sequential idx, wxyz)
vertex = zeros(N,8);
vertex(:,1)   = (1:N)';
vertex(:,2:4) = v_xyzq(:,1:3);
qxqyqzqw = v_xyzq(:,4:7);
vertex(:,5:8) = [qxqyqzqw(:,4), qxqyqzqw(:,1:3)]; % qw qx qy qz

% convert edges endpoints to row indices, store meas in wxyz + info
M = size(e_ij,1);
edge = zeros(M,30);
for k = 1:M
    ii = int64(e_ij(k,1));
    jj = int64(e_ij(k,2));
    if ~isKey(id2row, ii) || ~isKey(id2row, jj)
        error('Edge references missing vertex id (%d,%d).', e_ij(k,1), e_ij(k,2));
    end
    i = double(id2row(ii));
    j = double(id2row(jj));
    edge(k,1:2) = [i, j];
    edge(k,3:5) = e_meas(k,1:3);
    qxqyqzqw = e_meas(k,4:7);
    edge(k,6:9) = [qxqyqzqw(4), qxqyqzqw(1:3)];
    edge(k,10:30) = e_info(k,:);
end
end

function v = upper_tri_from_info6(M)
v = zeros(1,21);
idx = 1;
for i = 1:6
    for j = i:6
        v(idx) = M(i,j);
        idx = idx + 1;
    end
end
end
