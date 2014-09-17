function res = whiten_data(input, whM, useCuda)
    if (nargin < 3),
        useCuda = 0;
    end
    
    [H, W, colors, N] = size(input);
    
    w = sqrt(size(whM, 1) / colors);
    halfw = floor(w/2);
    
    res = zeros([H W colors N]);
    whMtemp = reshape(whM', [w, w, colors, w*w*colors]);
    exKernel = reshape(eye(w*w*colors), [w, w, colors, w*w*colors]);
    
    for i = 1:N,
        fprintf('%5d/%5d...',i,N);
        
        padded = zeros(H+w-1, W+w-1, colors);
        mu = mean(mean(input(:,:,:,i),1),2);
        
        padded(halfw:H+halfw-1, halfw:W+halfw-1, :) = bsxfun(@minus,input(:,:,:,i),mu);
        temp = convs(padded, whMtemp, useCuda);
        padded = conve(temp, exKernel, useCuda);
        
        res(:,:,:,i) = padded(halfw:H+halfw-1, halfw:W+halfw-1, :) ./ w^2;
        
        fprintf('done.\n');
    end
end

function y = evensign(x)
    if mod(x,2)
        y = -1;
    else
        y = 1;
    end
end