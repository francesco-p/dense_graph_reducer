% Sco 2014
% Generates a Graph (weigted graph) from a set of points X (nxd) and 
% a sigma decay. 
function [W] = GraphfromPoints(X,sigma)

n = size(X,1); 
W = zeros(n,n); 

for i=1:n 
    for j=1:n
        if i<j
            a = X(i,:); 
            b = X(j,:); 
            d = sum((a-b).^2); % To avoid ambiguities in dimension
            W(i,j) = exp(-sigma*d); 
        end
    end
end
% Symmetrize 
W = W+W';

            
