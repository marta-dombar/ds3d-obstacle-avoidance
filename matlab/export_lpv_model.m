% EXPORT_LPV_MODEL  Save the LPV-DS learned in gui_lpvDS.m so Python can load it.
%
% gui_lpvDS.m keeps A_k (handles.A_g) but discards b_k. Make this one-line
% change inside learn_DS_lpv_Callback, right after optimize_lpv_ds_from_data:
%
%     [A_k, b_k, ~] = optimize_lpv_ds_from_data(Data, att_g, constr_type, ds_gmm, P_opt, init_cvx);
%     save('lpv_model.mat', 'ds_gmm', 'A_k', 'b_k', 'att_g');      % <-- ADD THIS LINE
%
% Then: draw demonstrations, fit the GMM, press "Learn LPV-DS" and copy
% lpv_model.mat into the root of this repository. part1_2d.py and the 3D
% "lifted" DS will pick it up automatically.
%
% Expected contents (K = number of Gaussians):
%   ds_gmm.Priors 1xK, ds_gmm.Mu 2xK, ds_gmm.Sigma 2x2xK
%   A_k 2x2xK, b_k 2xK, att_g 2x1
%
% Optional consistency check (compare with Python's LPVDS.from_mat):
%   ds_lpv = @(x) lpv_ds(x, ds_gmm, A_k, b_k);
%   X = [0.1 -0.5; 0.3 0.2]';          % a few test points (columns)
%   disp(ds_lpv(X))
