%% Fast Poisson solver on an unit disk
%  with Dirichlet boundary conditions
%
%  Input:
%    M: number of grid points in r-direction (even)
%    N: number of grid points in theta-direction (must be an even number)
%    hh: Dirichlet boundary condition at r=1, function handle
%    f: laplace(u), function handle
%
%  OUTPUT:
%    u: numerical solution

%%
function u = poisson_solver_fd_disk_d(M, N, hh, f)
    %% Initialize variables
    dr = 1/M;
    r = ((1:M)-0.5)*dr;

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
    % Dirichlet bc at u(1)
    h = hh(theta);
    fft_h = fft(h);

    %% setting RHS
    ff = f(R, T);
    fft_f = fft(ff);

    %% construct the linear operaters
    A = spdiags((r-0.5*dr)', 1, M, M) ...
        + spdiags((r+0.5*dr)', -1, M, M) ...
        + spdiags(-2*r', 0, M, M);
    A(M,M) = A(M,M)-1;

    %% Solve the equation for each Fourier mode
    for ii = 1:N/2+1
        
        % setup of RHS
        RHS = dr^2*(r.*fft_f(ii,:)).';
        RHS(M) = RHS(M)-2*fft_h(ii);

        % combine the operators
        L = A - (dr^2)*(k_mode(ii)^2)*spdiags((1./r)', 0, M, M);

        %% solve the linear system to obtain solutions 
        fft_u(ii,:) = (L\RHS).';
    end
    
    %% take ifft to get the solution
    fft_u = [fft_u;conj(flipud(fft_u(2:N/2,:)))];
    u = real(ifft(fft_u));
end