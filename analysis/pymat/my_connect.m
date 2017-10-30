function A = my_connect(B)
    
    A = B;

    [S, C] = graphconncomp(sparse(A));

    display(S)

    if(S>1)
        min_weight = min(A(find(A)))/100;		
        cc_idx = unique(C);
        cc = cell(1,length(cc_idx));
        idx = 1:length(C);
        for k=1:length(cc_idx)
            cc{k} = idx(C==k);
        end
  
         for j=2:length(cc)
            A(cc{j-1}(end),cc{j}(1)) = min_weight;
            A(cc{j}(1),cc{j-1}(end)) = min_weight;
         end
    end    

    [S, C] = graphconncomp(sparse(A));
        
    display(S)

end