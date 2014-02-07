function compWhitMatrix(input, w)

samples = 100000;

N = size(input,4);
W = size(input,2);
H = size(input,1);

X = zeros(samples, w*w);
for i = 1:samples,
    im = randi(N);
    x = randi(W-w+1)-1; y = randi(H-w+1)-1;
    patch = input(y+1:y+w, x+1:x+w, :,im);
    X(i,:) = patch(:)';
end

[~,~,uwhM,whM]=whiten(X);

save(sprintf('whitM_%d',w), 'whM', 'uwhM');