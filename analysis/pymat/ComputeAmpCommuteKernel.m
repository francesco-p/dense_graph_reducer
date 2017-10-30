function ampDist = ComputeAmpCommuteKernel(W)
% computes the amplified commute kernel of a weigted undirected graph
% as defined in:
%
% Luxburg, Radl, Hein: Getting lost in space: Large sample analysis of the
% resistance distance, NIPS 2010
%
% Please cite this paper if you use this code
%
% input: symmetric, non-negative weight matrix W (can be full or sparse)
% output: the amplified commute kernel (which is positive semi-definite)
%
% this code is not optimized for sparse weight matrices !

if(abs(W-W')~=0)
  error(['Weight matrix has to be symmetric']);
end  

if(min(W(:))<0)
  error(['Weight matrix has to be non-negative']);
end  

c = GD_GetComps(W);
disp(['Graph has ',num2str(max(c)),' connected component(s)']);

if(max(c)>1)
 error(['Commute Time can only be computed for connected graph !']);
end

num=size(W,1);
d = sum(W,2); 
D = spdiags(d,0,num,num);

% generate Laplacian 
L = (D-W); 
clear D; %clear W;

% compute pseudoinverse PL of Laplacian 
first = 1/sqrt(num)*ones(num,1);
PL = inv(full(L)+first*first')-first*first'; % this is very inefficient
clear L;

% compute resistance distance cdist
PLD = diag(PL); first = ones(num,1);
cdist =(PLD*first' + first*PLD' - 2*PL); 
clear PLD; clear PL; 

% computation of amplified distance
invd=1./d; 
ampDist = cdist - invd*ones(num,1)' - ones(num,1)*invd';          % subtract limit
ampDist = ampDist - diag(diag(ampDist));                          % set diagonal to zero
CorrMatrix = spdiags(invd,0,num,num)*W*spdiags(invd,0,num,num);
DiagCorrMatrix = diag(CorrMatrix);

% amplified commute distance
ampDist = ampDist + 2*CorrMatrix - DiagCorrMatrix*ones(num,1)' - ones(num,1)*DiagCorrMatrix'; 
clear CorrMatrix; clear DiagCorrMatrix;

% define kernel from amplified commmute distance using centering
rowsumvec = ampDist*ones(num,1); totsum=sum(rowsumvec);
AmpKernel = -ampDist + (1/num*rowsumvec)*ones(num,1)' + (1/num*ones(num,1))*rowsumvec' - (totsum/num^2)*ones(num,num);
dK = diag(AmpKernel);

% normalize amplfified kernel
AmpKernel = spdiags(1./sqrt(dK),0,num,num)*AmpKernel*spdiags(1./sqrt(dK),0,num,num);

function [c] = GD_GetComps(A)
n = size(A,1);
c = zeros(n,1);
clNo = 1;
q = zeros(n,1);
qptr = 1;
qlen = 0;
for i = 1:n,
    if (c(i) == 0),
        c(i) = clNo;
        qlen = qlen + 1;
        q(qlen) = i;
        while (qptr <= qlen)
            j = q(qptr);

            nbrs = find(A(:,j));
            for nbr = nbrs';
                if (c(nbr) == 0),
                    qlen = qlen + 1;
                    q(qlen) = nbr;
                    c(nbr) = clNo;
                end
            end
            qptr = qptr + 1;
        end

        clNo = clNo + 1;
    end
end