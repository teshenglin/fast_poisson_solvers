function v = val2coef(u)
    N = length(u)-1;
    u = [u;flipud(u(2:N))];
    v = fft(u);
    v = v(1:N+1)/N;
    v(1) = 0.5*v(1);
    v(N+1) = 0.5*v(N+1); 
end