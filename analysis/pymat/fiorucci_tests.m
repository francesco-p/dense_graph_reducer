load('data/spect-singleproton.mat');
NG = my_connect(NG);
% ACT
NG_t = ComputeAmpCommuteKernel(NG);

% Normalization
NG_t = NG_t / max(NG_t(:));

% abs(NG_t - 1)
NG_t = abs(NG_t - 1);

% Distance 
d_norm = l2_dist(GT, NG_t)/numel(GT)

% Min e Max
maxel = max(NG_t(:))
minel = min(NG_t(:))

%load('data/spect-singleproton.mat');
%NG = my_connect(NG);
%% ACT
%NG_t = ComputeAmpCommuteKernel(NG);
%
%% Normalization
%NG_t = NG_t / max(NG_t(:));
%
%% Kernel sigma=1
%kNG_t = exp(-NG_t);
%
%% Distance
%d_norm = l2_dist(GT, kNG_t)/numel(GT)
%
%% Min Max elements
%maxel = max(kNG_t(:))
%minel = min(kNG_t(:))
%
