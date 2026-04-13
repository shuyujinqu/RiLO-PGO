
function [nbr, eidx] = Adjacency_Mat_row(A, i)
%ADJACENCY_MAT_ROW Outgoing neighbors and edge indices from node i.
[~, nbr, eidx] = find(A(i,:));
nbr = nbr(:);
eidx = eidx(:);
end
