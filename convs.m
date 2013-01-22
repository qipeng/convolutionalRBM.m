function Z = convs(X, Y, useCuda)
% CONVS  Shrinking matrix convolution in CRBM
%   Z = CONVS(X, Y, useCuda)
%       Takes X the n-by-n input image, Y the m-by-m convolutional filter,
%       returns the convolution result Z, which is (n-m+1)-by-(n-m+1)
%       
%       Set useCuda to 1 to use CUDA MEX files.
%
%       See also CONVE
%
%   Written by: Peng Qi, Sep 27, 2012

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

if ~isempty(useCuda) && useCuda,
    Z = convs_cuda(X, Y);
else
    Z = convsmex(X, Y);
end