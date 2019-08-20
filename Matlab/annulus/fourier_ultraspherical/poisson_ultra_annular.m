%% Fast spectral Poisson solver on annular domain
%
% Fast Poisson solver for
% laplace(u) = f on omega = {(x,y) = innerR^2 < x^2 + y^2 < outerR^2}
% with Fourier-Ultraspherical spectral method
%
% Rmk1: We assume innerR>0
%
% Rmk2: The boundary conditions are Dirichlet at innerR and outerR.
%       BCs are: u(innerR, theta) = g(theta)
%                u(outerR, theta) = h(theta)
%
% Example: 
%      u = sin(10*x)
%      f = -10^2*sin(10*x)
%      innerR = 0.5, outerR = 1.
%      Dirichlet bc at inner boundary
%                g = u(0.5, theta)= exp(0.5*(cos(theta) + sin(theta)))
%      Dirichlet bc at outer boundary
%                h = u(1,theta) = exp(cos(theta) + sin(theta))

%% Setup grid
% number of grid points in r-direction
M = 20;
% number of grid points in theta-direction
N = 100;

%% Setup domain
% omega = {(x,y) = 0 < innerR < r < outerR} 
innerR = 0.5;
outerR = 1;

X = @(R,T) R.*cos(T);
Y = @(R,T) R.*sin(T);

%% Setup the exact solution
%exact = @(R,T) exp(X(R,T) + Y(R,T));
exact = @(R,T) sin(10*X(R,T));

% f: right hand side of the equation
%f = @(R,T) 2*exp(X(R,T) + Y(R,T));
f = @(R,T) -10^2*exact(R,T);

% Setup boundary conditions
% Dirichlet bc at u(innerR)
gg = @(TT) exact(innerR, TT);

% Dirichlet bc at u(outerR)
hh = @(TT) exact(outerR, TT);

%% Fast spectral Poisson solver on annular domain
tic
u = poisson_solver_ultra_annular(M, N, innerR, outerR, gg, hh, f);
toc

%% In the following we check the accuracy of the numerical solutions

%% Grid points construction
% Chebyshev points
x = cos((0:M-1)*pi/(M-1));

% change valuable to grids in radial direction
alpha = 2/(outerR - innerR);
beta = (outerR + innerR)/(outerR - innerR);
r = (x + beta)/alpha;

% equal spaced nodes in polar direction
dtheta = 2*pi/N;
theta = (0:dtheta:(2*pi-dtheta));

% create 2D R-T grids
[R,T] = meshgrid(r, theta);

exact_sol = exact(R, T);

%% evaluate max error of the solution values
error = max(max(abs(u-exact_sol)));
disp(['error in L-\infty norm = ', num2str(error)])
