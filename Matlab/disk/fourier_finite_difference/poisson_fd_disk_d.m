%% Fast Poisson solver on an unit disk
%
% Fast Poisson solver for
% laplace(u) = f on omega = {(x,y) = 0 < x^2 + y^2 < 1}
% with Fourier-Finite difference method
%
% Rmk2: The boundary conditions are Dirichlet at R=1
%       BCs are: u(1, theta) = h(theta)
%
% Example: 
%      u = sin(10*x)
%      f = -10^2*sin(10*x)
%      Dirichlet bc:
%                h = u(1,theta) = sin(10*cos(theta))

%% Setup grid
% number of grid points in r-direction
M = 200;
% number of grid points in theta-direction
N = 100;

%% Setup domain
% omega = {(x,y) = 0 < r < 1} 

X = @(R,T) R.*cos(T);
Y = @(R,T) R.*sin(T);

%% Setup the exact solution
%exact = @(R,T) exp(X(R,T) + Y(R,T));
exact = @(R,T) sin(10*X(R,T));

% f: right hand side of the equation
%f = @(R,T) 2*exp(X(R,T) + Y(R,T));
f = @(R,T) -10^2*exact(R,T);

% Setup boundary conditions
% Dirichlet bc at u(r=1)
hh = @(TT) exact(1, TT);

%% Fast Poisson solver on an unit disk
tic
u = poisson_solver_fd_disk_d(M, N, hh, f);
toc

%% In the following we check the accuracy of the numerical solutions

%% Grid points construction
dr = 1/M;
r = ((1:M)-0.5)*dr;

% equal spaced nodes in polar direction
dtheta = 2*pi/N;
theta = (0:dtheta:(2*pi-dtheta));

% create 2D R-T grids
[R,T] = meshgrid(r, theta);

exact_sol = exact(R, T);

%% evaluate max error of the solution values
error = max(max(abs(u-exact_sol)));
disp(['error in L-\infty norm = ', num2str(error)])
