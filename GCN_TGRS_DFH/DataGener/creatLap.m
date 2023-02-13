function [W, D, L]=creatLap(X,k,sigma) 
      X=X';
      options = [];
      options.NeighborMode = 'KNN';
      options.k = k;
      options.WeightMode = 'HeatKernel';
      options.t = sigma;

      W = (constructW(X, options));
      D = (diag(sum(W, 2)));
      L = D - W;
end