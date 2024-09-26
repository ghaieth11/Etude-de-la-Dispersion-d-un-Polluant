close all; % Ferme toutes les figures ouvertes
figure; % Crée une nouvelle figure

nx = 40; % Nombre de points en x
ny = 40; % Nombre de points en y
kappa = 0.8; % Coefficient de diffusion
xmax = 10; % Valeur maximale de x
ymax = 10; % Valeur maximale de y
dx = xmax / (nx - 1); % Pas en x
dy = ymax / (ny - 1); % Pas en y
dt = 0.01; % Pas de temps

x = linspace(-xmax, xmax, nx); % Vecteur de x
y = linspace(-ymax, ymax, ny); % Vecteur de y
[X, Y] = meshgrid(x, y); % Grille de points

% Condition initiale
x0 = 2; 
y0 = 2; 
sigma = 0.8; 
C = CondInitiale(X, Y, sigma, x0, y0); 
Z = reshape(C, nx, ny); 

A = AssembleMatrix(nx, ny, dt, kappa, dx, dy, [1.5, 1.5], X, Y); % Assemblage de la matrice

% Préparation du tracé
h = surf(X, Y, Z); 
shading interp; 
colorbar; 
xlabel('X'); 
ylabel('Y'); 
zlabel('Concentration'); 
title('Évolution de la concentration'); 


% Boucle sur le temps
for t = 1:200
    disp(['t = ', num2str(t * dt)]); 

    % Mise à jour de la vitesse
    v(1) = 1.5 + 0.5 * sin(0.1 * t); 
    v(2) = 1.5 + 0.5 * cos(0.1 * t); 

    b = Rhs(nx, ny, dt, kappa, dx, dy, v, C, X, Y); % Calcul du RHS
    C = A \ b; % Résolution du système
    Z = reshape(C, nx, ny); % Reshape pour le tracé

    set(h, 'ZData', Z); % Mise à jour des données Z
    title(['Concentration à t = ', num2str(t * dt)]); % Mise à jour du titre


end
