function C = CondInitiale(X,Y,sigma,x0,y0)
% Initialisation d'une gaussienne 2D
C = exp(-((X - x0).^2 + (Y - y0).^2) / (2 * sigma^2));

end