clear all
close all


%% Loading true and acquired image
load('acquired_image.mat')
load('true_image.mat')
[M,N] = size(X);
Npix  = M*N;
%True image
figure()
imshow(X,[])
%Acquired image
figure()
imshow(Y,[])



%% Loading PSF
load('PSF.mat')
figure;
imagesc(psf);
colormap gray;
axis image;

%% 3) Implement D^T D
D =Fourier_D(M,N);

%% Gibbs Sampler

% Gibbs on (tau_e, tau_x, x)
alpha_e0 = 0; beta_e0 = 0; 
alpha_x0 = 0; beta_x0 = 0;

%Initialization
x = Y;                     
nIter  = 1000; 
BurnIn = 150; 
Thin = 2;
kept = {}; 
kidx = 0;
tauE = zeros(1,nIter);
tauX = zeros(1,nIter);

Hhat = psf2otf(psf, [M N]); 
for it = 1:nIter
    % (a) tau_e| x,y  ~ Gamma(N/2, ||y - Hx||^2 / 2)  via Parseval
    %Sampling from tau_e


    alpha_e = alpha_e0 + Npix/2;
    y_hat = fft2(Y);
    x_hat = fft2(x);
    res2 = sum(sum(abs(y_hat - Hhat .* x_hat).^2)) / Npix;   % ||y-Hx||^2 via Parseval
    beta_e = beta_e0 + res2/2;
    tau_e = rndgamma_shape_rate(alpha_e, beta_e);


    % (b) tau_x | x,y  ~ Gamma(N/2, ||x||_D^2 / 2)
    %Sampling tau_x
    xnormD2 = sum(sum(D .* abs(x_hat).^2)) / Npix;          % ||Dx||^2 via Parseval
    alpha_x = alpha_x0 + Npix/2;
    beta_x = beta_x0 + xnormD2/2;
    tau_x = rndgamma_shape_rate(alpha_x, beta_x);
  
    % (c) x | tau_e, tau_x, in the Fourier domain
    %Sampling x

    num = tau_e * conj(Hhat) .* y_hat;
    denom = tau_e * abs(Hhat).^2 + tau_x * D;
    muhat = num ./ denom;

    U = fft2(randn(M,N))/sqrt(Npix);%Scaled Gaussian white noise in Fourier domain

    x_hat_s = muhat + sqrt(Npix) * U ./ sqrt(denom);   % not X: X is the true image
    x = real(ifft2(x_hat_s));

    tauE(it) = tau_e;
    tauX(it) = tau_x;

    if it > BurnIn && mod(it - BurnIn, Thin) == 0
        kidx = kidx + 1; kept{kidx} = x; 
    end


     fprintf('it %d/%d | tau_e=%.2e, tau_x=%.2e\n', it, nIter, tau_e, tau_x);
end

% Evolution of the chains
figure();
subplot(3,1,1); plot(tauE); xline(BurnIn,'r--','Burn-in');
title('\tau_e'); xlabel('Iteration'); grid on;
subplot(3,1,2); plot(tauX); xline(BurnIn,'r--','Burn-in');
title('\tau_x'); xlabel('Iteration'); grid on;
subplot(3,1,3); plot(1./sqrt(tauE)); xline(BurnIn,'r--','Burn-in');
title('Noise std \sigma_e = 1/\surd\tau_e'); xlabel('Iteration'); grid on;

% Mean a posteriori
Xbar = zeros(M,N);
for k=1:kidx, Xbar = Xbar + kept{k}; end
Xbar = Xbar/kidx;  

% PSNR 
psnr_noisy = PSNR(X, Y);
psnr_post  = PSNR(X, Xbar);fprintf('PSNR noisy  : %.2f dB\n', psnr_noisy);
fprintf('PSNR posterior mean : %.2f dB (avec %d conversed sample)\n', psnr_post, kidx);

figure();
subplot(1,3,1); imshow(X,[]);    title('Original');
subplot(1,3,2); imshow(Y,[]);    title(sprintf('Noisy (PSNR %.1f dB)', psnr_noisy));
subplot(1,3,3); imshow(Xbar,[]); title(sprintf('Posterior mean (PSNR %.1f dB)', psnr_post));

%%
function g = rndgamma_shape_rate(alpha, rate)
if rate <= 0, rate = eps; end
try
    g = gamrnd(alpha, 1/rate);  % si Statistics Toolbox
catch
    g = randg(alpha) / rate;    % fallback de base
end
end

function p = PSNR(ref, est)
ref = im2double(ref); est = im2double(est);
mse = mean((ref(:)-est(:)).^2);
if mse == 0, p = 99; else, p = 20*log10(1/sqrt(mse)); end
end