
function [p0, t0] = init_from_odometry(vertex, edge)
%INIT_FROM_ODOMETRY Robust initialization by BFS over the graph.
% Uses edge measurements to propagate poses from node 1.
% Output:
%   p0: 4xN quaternions (wxyz)
%   t0: 3xN translations

N = size(vertex,1);
M = size(edge,1);

% adjacency lists (cell arrays)
adj = cell(N,1);
% Each neighbor row: [nbr, tx, ty, tz, qw, qx, qy, qz]  -> 1 + 3 + 4 = 8 columns
for k = 1:N, adj{k} = zeros(0,8); end

for e = 1:M
    i = edge(e,1); j = edge(e,2);
    tij = edge(e,3:5).';
    qij = edge(e,6:9).'; qij = qij / max(norm(qij),1e-12);

    % forward i->j
    adj{i}(end+1,:) = [j, tij.', qij.']; % [nbr tx ty tz qw qx qy qz]

    % inverse j->i
    Rij = quat2rotm_local(qij);
    qji = [qij(1); -qij(2:4)]; % conj
    tji = -(Rij.') * tij;
    adj{j}(end+1,:) = [i, tji.', qji.'];
end

p0 = zeros(4,N);
t0 = zeros(3,N);
visited = false(N,1);

% root at node 1
p0(:,1) = [1;0;0;0];
t0(:,1) = [0;0;0];
visited(1) = true;

% BFS queue
q = zeros(N,1); head=1; tail=1;
q(tail)=1;

while head<=tail
    u = q(head); head=head+1;
    Ru = quat2rotm_local(p0(:,u));
    tu = t0(:,u);
    neigh = adj{u};
    for kk = 1:size(neigh,1)
        v = neigh(kk,1);
        if visited(v), continue; end
        tij = neigh(kk,2:4).';
        qij = neigh(kk,5:8).'; qij = qij / max(norm(qij),1e-12);

        % compose
        pv = qmult_local(p0(:,u), qij);
        tv = tu + Ru * tij;

        p0(:,v) = pv / max(norm(pv),1e-12);
        t0(:,v) = tv;
        visited(v) = true;
        tail=tail+1; q(tail)=v;
    end
end

% Any disconnected nodes: fall back to initial from vertex if available, else identity
for i = 1:N
    if ~visited(i)
        p = vertex(i,5:8).'; 
        if numel(p)==4 && norm(p)>1e-12
            p0(:,i) = p / norm(p);
        else
            p0(:,i) = [1;0;0;0];
        end
        t0(:,i) = vertex(i,2:4).';
    end
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

function q = qmult_local(q1,q2)
% quaternion multiply (wxyz)
w1=q1(1); x1=q1(2); y1=q1(3); z1=q1(4);
w2=q2(1); x2=q2(2); y2=q2(3); z2=q2(4);
q = [w1*w2 - x1*x2 - y1*y2 - z1*z2;
     w1*x2 + x1*w2 + y1*z2 - z1*y2;
     w1*y2 - x1*z2 + y1*w2 + z1*x2;
     w1*z2 + x1*y2 - y1*x2 + z1*w2];
end
