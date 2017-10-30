load('data/iris.mat');
%load('data/ionosphere.mat');

% Creates "not"-GT and puts 0 on diagonal
%nGT = ~GT;
%nGT(logical(eye(size(GT)))) = 0;
%nGT = double(nGT);
%nGT = nGT*0.00001

% Line graph
%nGT = tril(ones(size(GT)),-1) - tril(ones(size(GT)), -2);
%nGT = nGT + nGT';
%nGT = double(nGT);
%nGT = nGT*0.00001

% Star graph
%nGT = zeros(size(GT));
%nGT(:,1) = 1;
%nGT(1,:) = 1;
%nGT(1,1) = 0;

%imshow(nGT)

% Connect all the graphs
%GT = my_connect(GT);
NG = my_connect(NG);
%nGT = my_connect(nGT);

% Compute Amplitude commute Time
%GT = ComputeAmpCommuteKernel(GT);
NG_t = ComputeAmpCommuteKernel(NG);
%nGT = ComputeAmpCommuteKernel(nGT);

% Normalization
NG_t = NG_t / max(NG_t(:));

NG_t = NG_t - 1;
NG_t = abs(NG_t);

% Apply gaussian kernel
kNG_t = exp(-NG_t/sigma);
%kNG_t = exp(-NG_t);


%max_d = l2_dist(GT, nGT);
d = l2_dist(GT, kNG_t);

%display(max_d);
display(d/numel(GT));