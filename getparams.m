function params = getparams()
% GETPARAMS  Get default params for trainCRBM
%
%   See also TRAINRBM
%
%   Written by: Peng Qi, Sep 27, 2012

%% Model parameters
params.nmap = 10;
params.szFilter = 5;
params.szPool = 3;
% params.stpFilter = 1;
% params.stpPool = 1;

%% Learining parameters
params.epshbias = 1e-2;
params.epsvbias = 1e-2;
params.epsW = 1e-3;
params.phbias = 0.5;
params.pvbias = 0.5;
params.pW = 0.5;
params.decayw = 1e-2;
params.szBatch = 1;
params.sparseness = .01;

%% Running parameters
params.iter = 1000;
params.verbose = 1;
params.mfIter = 5;
params.saveInterv = 5;
params.useCuda = 0;
params.saveName = 'model.mat';

end