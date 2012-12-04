function [X] = conve(Z, Y, useCuda)
% CONVE  Expanding matrix convolution in CRBM
%   X = CONVE(Z, Y)
%       Takes Z the nz-by-nz input image, Y the m-by-m convolutional filter,
%       returns the convolution result X, which is (nz+m-1)-by-(nz+m+1)
%
%       See also CONVS
%
%   Written by: Peng Qi, Sep 27, 2012

if (size(Z,1) ~= size(Z,2) || size(Y,1) ~= size(Y,2))
    error('Matrices Z and Y should be square matrices.');
end

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

% nz = size(Z,1);
% m = size(Y,1);
% 
% n = nz+m-1;
% X = zeros(n);
% 
% for i = 1:nz,
%     for j = 1:nz,
%         X(i:i+m-1, j:j+m-1) = X(i:i+m-1, j:j+m-1) + Y .* Z(i,j);
%     end
% end

if ~isempty(useCuda) && useCuda,
    X = convd_cuda(Z, Y);
else
    X = convemex(Z, Y);
end