function params = getparams()
% GETPARAMS  Get default params for trainCRBM
%
%   See also TRAINRBM
%
%   Written by: Peng Qi, Sep 27, 2012

%% Model parameters
params.nmap = 20;
params.szFilter = 5;
params.szPool = 3;

%% Learining parameters
params.epshbias = 1e-1;
params.epsvbias = 1e-1;
params.epsW = 1e-2;
params.phbias = 0.5;
params.pvbias = 0.5;
params.pW = 0.5;
params.decayw = 1;
params.szBatch = 10;
params.sparseness = .01;

%% Running parameters
params.iter = 100;
params.verbose = 1;
params.mfIter = 5;
params.saveInterv = 5;
params.useCuda = 0;
params.saveName = 'model.mat';

end