%% Fast spectral Poisson solver on annular domain 
%  with Dirichlet boundary conditions
%
%  Input:
%    M: number of grid points in r-direction (even)
%    N: number of grid points in theta-direction (must be an even number)
%    innerR, outerR: inner and outer radius of the annulus
%    gg: Dirichlet boundary condition at innerR, function handle
%    hh: Dirichlet boundary condition at outerR, function handle
%    f: laplace(u), function handle
%
%  OUTPUT:
%    u: numerical solution

%%
function u = poisson_solver_ultra_annular(M, N, innerR, outerR, gg, hh, f)
    %% Initialize variables
    % Chebyshev points
    x = cos((0:M-1)*pi/(M-1));

    % change valuable in radial direction
    alpha = 2/(outerR - innerR);
    beta = ((outerR + innerR)/(outerR - innerR));
    r = (x + beta)/alpha;      

    % valuable in polar direction
    dtheta = 2*pi/N;
    theta = (0:dtheta:(2*pi-dtheta));

    % k_mode: fourier mode
    k_mode = [0,1:N/2, -N/2+1:-1]';

    % the overall mesh grids
    [R,T] = meshgrid(r, theta);

    % fft_u: Fourier coefficients of u
    fft_u = zeros(N/2+1,M);

    %% boundary conditions
    % Dirichlet bc at u(innerR)
    g = gg(theta);
    fft_g = fft(g);

    % Dirichlet bc at u(outerR)
    h = hh(theta);
    fft_h = fft(h);

    %% setting RHS
    ff = f(R, T);
    fft_f = fft(ff);

    %% construct the linear operaters
    % differentiation operators
    D_1 = spdiags((0:M-1).', 1, M, M);
    D_2 = 2*spdiags((0:M-1).', 2, M, M);

    % conversion operators
    S_0 = spdiags([1 0.5*ones(1,M-1)].',0, M, M) ...
        + spdiags(-0.5*ones(1,M).', 2, M, M);
    S_1 = spdiags((1./(1:M)).',0, M, M) ...
        + spdiags([0 0 (-1./(3:M+3))].', 2, M, M);
    
    % Multiplication operators
    M_1 = beta*speye(M) + spdiags(0.5*ones(1,M).', 1, M, M) ...
        + spdiags(0.5*ones(1,M).', -1, M, M);  

    % M2(j,j)
    v1 = beta^2 + 1/6 + ((0:M-1).*(4:M+3))./(3*(1:M).*(3:M+2));
    % M2(j,j+1)
    v2 = [0 (4:M+2)./(3:M+1)]*beta;
    % M2(j,j+2)
    v3 = [0 0 (5:M+2)./(4*(3:M))];
    % M2(j+1,j)
    v4 = beta*(1:M)./(2:M+1);
    % M2(j+2,j)
    v5 = (1:M)./(4*(3:M+2));
    M_2 = spdiags(v1.', 0, M, M) + spdiags(v2.', 1, M, M) ...
        + spdiags(v3.', 2, M, M) + spdiags(v4.', -1, M, M) ...
        + spdiags(v5.', -2, M, M);
    
    %% boundary condition
    % Dirichlet boundary condition on the left
    bc_dl = ones(1, M); bc_dl(2:2:end) = -bc_dl(2:2:end);
    % Dirichlet boundary condition on the right
    bc_dr = ones(1, M);
    
    %% take the projection operators
    % reconstruct S1S0 so that the first two rows are empty
    S1S0 = S_1*S_0;
    [rows1s0,cols1s0,vals1s0] = find(S1S0);
    ind = rows1s0> (M-2);
    rows1s0(ind) = [];
    cols1s0(ind) = [];
    vals1s0(ind) = [];
    rows1s0 = rows1s0+2;
    S1S0 = sparse(rows1s0, cols1s0, vals1s0);

    % reconstruct L0 so that the first two rows are empty
    L_0 = M_2*D_2 + S_1*M_1*D_1;
    [rowl0,coll0,vall0] = find(L_0);
    ind = rowl0> (M-2);
    rowl0(ind) = [];
    coll0(ind) = [];
    vall0(ind) = [];
    rowl0 = rowl0+2;
    L_0 = sparse(rowl0, coll0, vall0);

    %% initialize U for woodbury formula
    U = zeros(M,2); U(1,1) = 1; U(2,2) = 1;

    %% Solve the equation for each Fourier mode
    for ii = 1:N/2+1
        
        % setup of RHS
        F = r.^2.*fft_f(ii,:);
        ff = val2coef(F.');
        % RHS
        RHS = [fft_g(ii); fft_h(ii); S1S0(3:end,:)*ff];

        % combine the operators
        L = L_0 - (k_mode(ii)^2)*S1S0;

        %% Woodbury formula to solve the linear system
        % (L + UV)^(-1) = ...
        
        % setup of LHS
        L(1,1:3) = bc_dl(1:3);
        L(2,1:4) = bc_dr(1:4);
        V = [zeros(1,3) bc_dl(4:M); zeros(1,4) bc_dr(5:M)];

        %% solve the linear system to obtain coeff. of solution 
        y = L\[RHS U];
        y2 = (eye(2) + V*y(:,2:3))\V;
        c = y(:,1) - y(:,2:3)*(y2*y(:,1));

        %% change the coeff to value
        fft_u(ii,:) = coef2val(c).';
    end
    
    %% take ifft to get the solution
    fft_u = [fft_u;conj(flipud(fft_u(2:N/2,:)))];
    u = real(ifft(fft_u));
end