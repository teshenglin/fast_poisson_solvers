function u = coef2val(v)
    N = length(v)-1;
    %Construct the corresponding coefficients of Fourier Series
    v = N*v;
    v = [v;flipud(v(2:N))];
    v(1) = 2*v(1);
    v(N+1) = 2*v(N+1);
    u = ifft(v);
    u = u(1:N+1);
end