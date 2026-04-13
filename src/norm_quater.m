function e = norm_quater(Q1, Q2)
%NORM_QUATER Frobenius-like quaternion difference with sign ambiguity handling.
% Q1, Q2: N-by-4, quaternion rows in [qw qx qy qz].
if isempty(Q1) || isempty(Q2)
    e = NaN;
    return;
end
if size(Q1,2) ~= 4 || size(Q2,2) ~= 4 || size(Q1,1) ~= size(Q2,1)
    error('norm_quater expects two N-by-4 quaternion arrays of the same size.');
end
Q1 = Q1 ./ max(vecnorm(Q1,2,2), 1e-12);
Q2 = Q2 ./ max(vecnorm(Q2,2,2), 1e-12);
err1 = Q1 - Q2;
err2 = Q1 + Q2; % q and -q are equivalent
err = min(sum(err1.^2,2), sum(err2.^2,2));
e = sqrt(sum(err));
end
