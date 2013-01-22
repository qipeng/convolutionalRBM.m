function Z = convs4(X,Y, useCuda)
% CONVS4  Shrinking matrix convolution in CRBM for weights calculation
%   Z = CONVS4(X, Y, useCuda)
%       Takes X the n-by-n input image, Y the m-by-m convolutional filter,
%       returns the convolution result Z, which is (n-m+1)-by-(n-m+1)
%       
%       Set useCuda to 1 to use CUDA MEX files.
%
%
%   Written by: Peng Qi, Jan 12, 2013

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

if ~isempty(useCuda) && useCuda,
    Z = convs4_cuda(X, Y);
else
    Z = convs4mex(X, Y);
end