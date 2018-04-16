function L = spChol(A)
% Cholesky decomposition for sparse band-like matrix A
% spChol(A) is an alternative to chol(A,'lower')
% L = spChol(A) returns the lower triangular matrix L in a special matrix format
% Note that size(L) = length(A)*(maxDiagIdOfA+1) !!!
% 1st column of L represents the zero diagonal of lower triangular matrix L
% other columns of L represents nonzero subdiagonals consequtively

  [~,pd] = chol(A);       % 0 ... pos def matrix

  if ~logical(pd)

    [~,q] = spdiags(A);   % returns indices of non-zero diagonals
    q = max(q);

    L = zeros(length(A), q+1);     % initialize quasi-sparse lower triangular matrix of Cholesky decomposition

    for i=1:length(A)

      ixm = min(q+1,i);
      sumL2jk = 0;        % reset sum of squares of Ljk for forthcoming Lj,j calculation
         
      for j=1:ixm-1
        
        jr = max(i-q-1,0) + j;
        jy = ixm + 1 - j;
        sumLikLjk = 0;    % reset sum of squares of Lik*Ljk 
        
        for k = j-1:-1:1
          jrk = jr - k;   % for Lik as well as Ljk
          jyi = jy + k;   % Lik
          jyj = 1  + k;   % Ljk
          sumLikLjk = sumLikLjk  + L(jrk,jyi) * L(jrk,jyj);
        end
        
        L(jr,jy) = 1/L(jr,1)  * (A(i,jr) - sumLikLjk);
        sumL2jk = sumL2jk + L(jr,jy)*L(jr,jy);
        
      end
      
      L(i,1) = sqrt(A(i,i) - sumL2jk);
     
    end
    
  else error('NOT a definite positive matrix!'); 
  end
  
end