function A = AssembleMatrix(nx, ny, dt, kappa, dx, dy, v, X, Y)
    N = nx * ny; % Taille totale du système
    A = sparse(N, N); % Matrice creuse (économique en mémoire)
    
    % Coefficients des termes de diffusion
    rx = kappa * dt / dx^2;
    ry = kappa * dt / dy^2;
    
    % Coefficients des termes de convection
    vx = v(1);
    vy = v(2);
    cx = vx * dt / (2 * dx);
    cy = vy * dt / (2 * dy);

    % Assemblage de la matrice A avec la méthode des différences finies
    for j = 1:ny
        for i = 1:nx
            idx = (j-1)*nx + i; % Index global pour le point (i,j)
            
            % Diffusion
            if i > 1
                A(idx, idx-1) = -rx;  % Point de gauche
            end
            if i < nx
                A(idx, idx+1) = -rx;  % Point de droite
            end
            if j > 1
                A(idx, idx-nx) = -ry;  % Point en bas
            end
            if j < ny
                A(idx, idx+nx) = -ry;  % Point en haut
            end
            % Convection
            if i > 1
                A(idx, idx-1) = A(idx, idx-1) + cx;
            end
            if i < nx
                A(idx, idx+1) = A(idx, idx+1) - cx;
            end
            if j > 1
                A(idx, idx-nx) = A(idx, idx-nx) + cy;
            end
            if j < ny
                A(idx, idx+nx) = A(idx, idx+nx) - cy;
            end

            % Terme central
            A(idx, idx) = 1 + 2*rx + 2*ry;
        end
    end
end
