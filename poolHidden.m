function [hidres, poolres, hidsample] = poolHidden(poshidacts, hbias, p, useCuda)
% POOLHIDDEN  Computing the pooling results (samples) for a CRBM
%   [hidres, poolres, hidsample] = POOLHIDDEN(poshidacts, hbias, p, useCuda)
%       poshidacts  Activations for hidden variables
%       hbias       Biases for hidden variables
%       p           Pooling size
%       
%       Set useCuda to 1 to use CUDA MEX files.
%
%       See also CONVS
%
%   Written by: Peng Qi, Jan 12, 2013

if ~exist('useCuda', 'var') || isempty(useCuda),
    useCuda = 0;
end

if ~isempty(useCuda) && useCuda,
    if nargout == 3,
        [hidres, poolres, hidsample] = poolH_cuda(poshidacts, hbias, p);
    else
        [hidres, poolres] = poolH_cuda(poshidacts, hbias, p);
    end
else
    if nargout == 3,
        [hidres, poolres, hidsample] = poolHmex(poshidacts, hbias, p);
    else
        [hidres, poolres] = poolHmex(poshidacts, hbias, p);
    end
end