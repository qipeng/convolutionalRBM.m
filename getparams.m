function params = getparams(method)
% GETPARAMS  Get default params for trainCRBM
%
%   See also TRAINCRBM
%
%   Written by: Peng Qi, Sep 27, 2012

%% Model parameters
params.nmap = 16;
params.szFilter = 8;
params.szPool = 2;
params.method = 'CD';

if (nargin > 0)
    if strcmp(method, 'PCD'),
        params.method = 'PCD';
    end
end

%% Learining parameters
params.epshbias = 1e-1;
params.epsvbias = 1e-1;
params.epsW = 1e-2;
params.phbias = 0.5;
params.pvbias = 0.5;
params.pW = 0.5;
params.decayw = .01;
params.szBatch = 10;
params.sparseness = .02;
params.whitenData = 1;

%% Running parameters
params.iter = 10000;
params.verbose = 2;
params.mfIter = 5;
params.saveInterv = 5;
params.useCuda = 0;
params.saveName = 'model.mat';

end