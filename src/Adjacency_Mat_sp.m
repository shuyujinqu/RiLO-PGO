
function A = Adjacency_Mat_sp(vertex, edge)
%ADJACENCY_MAT_SP Sparse adjacency with edge indices as values.
% A(i,j) = e means edge e is from i=edge(e,1) to j=edge(e,2).
N = size(vertex,1);
M = size(edge,1);
i = edge(:,1); j = edge(:,2);
val = (1:M).';
A = sparse(i, j, val, N, N);
end
