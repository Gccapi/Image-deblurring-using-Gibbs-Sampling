function D_tild = Fourier_D(M, N)
    Dx_tild = fft2([1 -1], M, N);
    Dy_tild = fft2([1; -1], M, N);
    D_tild  = abs(Dx_tild).^2 + abs(Dy_tild).^2;
end