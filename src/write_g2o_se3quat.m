function write_g2o_se3quat(out_file, vertex, edge, ids)
%WRITE_G2O_SE3QUAT Write standard VERTEX_SE3:QUAT / EDGE_SE3:QUAT g2o.

if nargin < 4 || isempty(ids)
    ids = (1:size(vertex,1)).';
end
assert(numel(ids) == size(vertex,1), 'ids length mismatch');

fid = fopen(out_file,'w');
assert(fid >= 0, 'Cannot open output g2o file: %s', out_file);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

% vertices: x y z qx qy qz qw
for i = 1:size(vertex,1)
    x = vertex(i,2); y = vertex(i,3); z = vertex(i,4);
    qw = vertex(i,5); qx = vertex(i,6); qy = vertex(i,7); qz = vertex(i,8);
    fprintf(fid, 'VERTEX_SE3:QUAT %d %.16g %.16g %.16g %.16g %.16g %.16g %.16g\n', ...
        ids(i), x, y, z, qx, qy, qz, qw);
end

for k = 1:size(edge,1)
    i = ids(edge(k,1));
    j = ids(edge(k,2));
    tx = edge(k,3); ty = edge(k,4); tz = edge(k,5);
    qw = edge(k,6); qx = edge(k,7); qy = edge(k,8); qz = edge(k,9);
    info = edge(k,10:30);
    fprintf(fid, 'EDGE_SE3:QUAT %d %d %.16g %.16g %.16g %.16g %.16g %.16g %.16g', ...
        i, j, tx, ty, tz, qx, qy, qz, qw);
    for m = 1:numel(info)
        fprintf(fid, ' %.16g', info(m));
    end
    fprintf(fid, '\n');
end
end
