
function [nbr, eidx] = Adjacency_Mat_col(A, i)
%ADJACENCY_MAT_COL Incoming neighbors and edge indices into node i.
[nbr, ~, eidx] = find(A(:,i));
nbr = nbr(:);
eidx = eidx(:);
end
