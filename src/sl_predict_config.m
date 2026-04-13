
function cfg = sl_predict_config(model, feat)
%SL_PREDICT_CONFIG Predict solver hyperparameters from a trained MLP model.
% model: struct produced by sl_train_mlp (fields: W1,b1,W2,b2,muX,sigX,muY,sigY,param_names, param_bounds)
% feat: 1xD vector

x = feat(:);
x = (x - model.muX(:)) ./ model.sigX(:);

% forward
h = model.W1 * x + model.b1;
h = max(h, 0); % ReLU
y = model.W2 * h + model.b2;
y = y .* model.sigY(:) + model.muY(:);

% decode targets (see sl_train_mlp)
% y = [log(beta), log(rho-1), log(max_beta), aa_m, aa_damp, stage1, stage2, log(kappa0)]
beta = exp(y(1));
rho = 1 + exp(y(2));
max_beta = exp(y(3));
aa_m = round(y(4));
aa_damping = y(5);
stage1_iter = round(y(6));
stage2_iter = round(y(7));
kappa0 = exp(y(8));

% apply bounds
b = model.param_bounds;
beta = clamp(beta, b.beta(1), b.beta(2));
rho  = clamp(rho,  b.rho(1),  b.rho(2));
max_beta = clamp(max_beta, b.max_beta(1), b.max_beta(2));
aa_m = round(clamp(aa_m, b.aa_m(1), b.aa_m(2)));
aa_damping = clamp(aa_damping, b.aa_damping(1), b.aa_damping(2));
stage1_iter = round(clamp(stage1_iter, b.stage1_iter(1), b.stage1_iter(2)));
stage2_iter = round(clamp(stage2_iter, b.stage2_iter(1), b.stage2_iter(2)));
kappa0 = clamp(kappa0, b.kappa0(1), b.kappa0(2));

cfg = struct();
cfg.beta = beta;
cfg.rho = rho;
cfg.max_beta = max_beta;
cfg.aa_m = aa_m;
cfg.aa_damping = aa_damping;
cfg.stage1_iter = stage1_iter;
cfg.stage2_iter = stage2_iter;
cfg.vmf_kappa0 = kappa0;
end

function v = clamp(v, a, b)
v = max(a, min(b, v));
end
