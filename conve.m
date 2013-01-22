function [X] = conve(Z, Y, useCuda)
% CONVE  Expanding matrix convolution in CRBM
%   X = CONVE(Z, Y)
%       Takes Z the nz-by-nz input image, Y the m-by-m convolutional filter,
%       returns the convolution result X, which is (nz+m-1)-by-(nz+m+1)
%       
%       Set useCuda to 1 to use CUDA MEX files.
%
%       See also CONVS
%
%   Written by: Peng Qi, Sep 27, 2012

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

if ~isempty(useCuda) && useCuda,
    X = conve_cuda(Z, Y);
else
    X = convemex(Z, Y);
end