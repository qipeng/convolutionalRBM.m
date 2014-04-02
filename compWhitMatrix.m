function compWhitMatrix(input, w)

samples = 100000;

[H, W, C, N] = size(input);

X = zeros(samples, w*w*C);
for i = 1:samples,
    im = randi(N);
    x = randi(W-w+1)-1; y = randi(H-w+1)-1;
    patch = input(y+1:y+w, x+1:x+w, :,im);
    X(i,:) = patch(:)';
end

[~,~,uwhM,whM]=whiten(X);

save(sprintf('whitM_%d_%d',w,C), 'whM', 'uwhM');