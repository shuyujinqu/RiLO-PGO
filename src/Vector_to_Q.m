function q = Vector_to_Q(v)
%VECTOR_TO_Q Embed a 3-vector into a pure quaternion [0; v].
v = v(:);
if numel(v) ~= 3
    error('Vector_to_Q expects a 3-vector.');
end
q = [0; v];
end
