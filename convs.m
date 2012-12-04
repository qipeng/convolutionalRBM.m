function Z = convs(X, Y, useCuda, separate)
% CONVS  Shrinking matrix convolution in CRBM
%   Z = CONVS(X, Y)
%       Takes X the n-by-n input image, Y the m-by-m convolutional filter,
%       returns the convolution result Z, which is (n-m+1)-by-(n-m+1)
%
%       See also CONVE
%
%   Written by: Peng Qi, Sep 27, 2012

if (size(X,1) ~= size(X,2) || size(Y,1) ~= size(Y,2))
    error('Matrices X and Y should be square matrices.');
end

n = size(X,1);
m = size(Y,1);

if (m > n),
    error('Convolutional filter Y should be smaller than input X.');
end

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

if ~exist('separate', 'var') || isempty(separate),
    separate = 0;
end

% nz = n-m+1;
% Z = zeros(nz);
% 
% for i = 1:nz,
%     for j = 1:nz,
%         Z(i,j) = sum(sum(Y .* X(i:i+m-1, j:j+m-1)));
%     end
% end

if ~isempty(useCuda) && useCuda,
    Z = convs_cuda(X, Y, separate);
else
    Z = convsmex(X, Y, separate);
end