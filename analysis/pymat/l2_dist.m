function d = l2_dist(GT, NG)
diff = GT - NG;
diff = diff.^2;
dist2 = sum(sum(diff)); 
d = sqrt(dist2)

end

