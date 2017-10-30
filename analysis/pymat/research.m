GT = noisyblockadjmat(2,100);

% Creates "not"-GT and puts 0 on diagonal
%nGT = ~GT;
%nGT(logical(eye(size(GT)))) = 0;
%nGT = double(nGT);
%nGT = double(nGT)

% Line graph
nGT = tril(ones(size(GT)),-1) - tril(ones(size(GT)), -2);
nGT = nGT + nGT';
nGT = nGT*10^-9;

% Fiorucci inverse
L = nGT;
aux = GT;
aux(logical(eye(size(aux)))) = 1;
aux = flipud(aux);
nGT = aux + L;

% Star graph
%nGT = zeros(size(GT));
%nGT(:,1) = 1;
%nGT(1,:) = 1;
%nGT(1,1) = 0;

%imshow(nGT)

% Connect all the graphs
GT = my_connect(GT);
nGT = my_connect(nGT);

% Compute Amplitude commute Time
GT = ComputeAmpCommuteKernel(GT);
nGT = ComputeAmpCommuteKernel(nGT);

figure;
imshow(GT/max(GT(:)))
figure;
imshow(nGT/max(nGT(:)))

max_d = l2_dist(GT, nGT);

display(max_d);
