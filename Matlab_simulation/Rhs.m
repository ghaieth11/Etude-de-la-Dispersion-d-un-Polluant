function b = Rhs(nx, ny, dt, kappa, dx, dy, v, C, X, Y)
    N = nx * ny; % Taille totale du système
    b = zeros(N, 1);
    
    % Coefficients des termes de diffusion
    rx = kappa * dt / dx^2;
    ry = kappa * dt / dy^2;
    
    % Coefficients des termes de convection
    vx = v(1);
    vy = v(2);
    cx = vx * dt / (2 * dx);
    cy = vy * dt / (2 * dy);

    % Construction du second membre
    for j = 1:ny
        for i = 1:nx
            idx = (j-1)*nx + i; % Index global pour le point (i,j)
            Cij = C(idx); % Valeur de C au point (i,j)
            
            % Diffusion (schéma centré)
            Dij = 0;
            if i > 1
                Dij = Dij + rx * C(idx-1);
            end
            if i < nx
                Dij = Dij + rx * C(idx+1);
            end
            if j > 1
                Dij = Dij + ry * C(idx-nx);
            end
            if j < ny
                Dij = Dij + ry * C(idx+nx);
            end
            
            % Convection (schéma décentré en amont)
            Cconv = 0;
            if i > 1
                Cconv = Cconv - cx * C(idx-1);
            end
            if i < nx
                Cconv = Cconv + cx * C(idx+1);
            end
            if j > 1
                Cconv = Cconv - cy * C(idx-nx);
            end
            if j < ny
                Cconv = Cconv + cy * C(idx+nx);
            end
            
            % Assemblage du terme de droite
            b(idx) = Cij + Dij + Cconv;
        end
    end
end
