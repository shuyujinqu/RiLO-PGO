
function W = matrixW(q)
%MATRIXW Right quaternion multiplication matrix for wxyz quaternions.
% For p (wxyz), p ⊗ q = matrixW(q) * p
q = q(:);
if numel(q)~=4, error('q must be 4x1'); end
w=q(1); x=q(2); y=q(3); z=q(4);
W = [ w, -x, -y, -z;
      x,  w,  z, -y;
      y, -z,  w,  x;
      z,  y, -x,  w ];
end
