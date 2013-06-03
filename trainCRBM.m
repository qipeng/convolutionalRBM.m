function [model output] = trainCRBM(data, params, oldModel)
% TRAINCRBM  Trains a convolutional restricted Boltzmann machine 
%   with the specified parameters.
%
%   [model output] = TRAINCRBM(data, params, oldModel)
%
%   data should be a structure, containing:
%       data.x      The input images / pooling states of the previous layer
%                   of CRBM. This matrix is 4-D, where the first dimension
%                   indexes through the images, and the last three
%                   dimensions define an image (channels, x, y)
%
%   Written by: Peng Qi, Sep 27, 2012
%   Version: 0.2 alpha

if params.verbose > 0,
    fprintf('Starting training CRBM with the following parameters:\n');
    disp(params);
    fprintf('Initializing parameters...');
end

useCuda = params.useCuda;

if isfield(params, 'method'),
    if strcmp(params.method, 'CD'),
        method = 1;
    elseif strcmp(params.method, 'PCD'),
        method = 2;
    end
else
    method = 1;     % use CD as default
end

%% initialization
N = size(data.x, 1);
K = params.nmap;
m = params.szFilter;
p = params.szPool;
n = size(data.x, 3);
K0 = size(data.x, 2);
nh = n - m + 1;
np = floor(nh / p);
param_iter = params.iter;
param_szBatch = params.szBatch;
output_enabled = nargout > 1;

%vmask = conve(ones(nh), ones(m), useCuda);

hinit = 0;

if params.sparseness > 0,
    hinit = -1;
end

if exist('oldModel','var') && ~isempty(oldModel),
    model = oldModel;
    if (~isfield(model,'W')), 
        model.W = 0.01 * randn(K0, K, m, m);
    else
        if (size(model.W) ~= [K0 K m m]), error('Incompatible input model.'); end
    end
    if (~isfield(model,'vbias')), model.vbias = zeros(1, K0);end
    if (~isfield(model,'hbias')), model.hbias = ones(1, K) * hinit;end
    if (~isfield(model,'sigma')),
        if (params.sparseness > 0)
            model.sigma = 0.4;
        else
            model.sigma = 1;    
        end
    end
else
    model.W = 0.01 * randn(K0, K, m, m);
    model.vbias = zeros(1, K0);
    model.hbias = ones(1, K) * hinit;
    if (params.sparseness > 0)
        model.sigma = 0.05;
    else
        model.sigma = 1;    
    end
end

dW = 0;
dvbias = 0;
dhbias = 0;

pW = params.pW;
pvbias = params.pvbias;
phbias = params.phbias;

if output_enabled,
    output.x = zeros(N, K, np, np);
end

total_batches = floor(N / param_szBatch);

if params.verbose > 0,
    fprintf('Completed.\n');
end

hidq = params.sparseness;
lambdaq = 0.9;

if ~isfield(model,'iter')
    model.iter = 0;
end

if (params.whitenData),
    if (params.verbose > 0), fprintf('Whitening data...'); end
%     try
%         load(sprintf('whitM_%d', params.szFilter));
%     catch e,
%         compWhitMatrix(params.szFilter);
%         load(sprintf('whitM_%d', params.szFilter));
%     end
    data.x = whiten_data(data.x, useCuda);
    if (params.verbose > 0), fprintf('Completed.\n'); end
end

if method == 2,
    phantom = randn(N, K0, n, n);
end

for iter = model.iter+1:param_iter,
    % shuffle data
    batch_idx = randperm(N);
    
    if params.verbose > 0,
        fprintf('Iteration %d\n', iter);
        if params.verbose > 1,
            fprintf('Batch progress (%d total): ', total_batches);
        end
    end
    
    hidact = zeros(1, K);
    errsum = 0;
    
    if (iter > 10),
        params.pW = .9;
        params.phbias = .9;
        params.pvbias = .9;
    end
    
    for batch = 1:total_batches,
        batchdata = data.x(batch_idx((batch - 1) * param_szBatch + 1 : ...
            batch * param_szBatch),:,:,:);
        if method == 2,
            phantomdata = phantom(batch_idx((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch),:,:,:);
        end
        recon = batchdata;
        
        %% positive phase

        %% mean field hidden update
        
        model_W = model.W;
        model_hbias = model.hbias;
        model_vbias = model.vbias;
        
        poshidacts = reshape(sum(convs(recon, model_W, useCuda),2), [param_szBatch K nh nh]);

        [poshidprobs, pospoolprobs, poshidstates] = poolHidden(poshidacts, model_hbias, p, useCuda);
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output_x = pospoolprobs;
        end
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output.x((batch - 1) * param_szBatch + 1:batch * param_szBatch,:,:,:) = output_x;
        end
        
        %% negative phase
        
        %% reconstruct data from hidden variables

        if method == 1,
            recon = reshape(sum(conve(poshidstates, model_W, useCuda), 3), [param_szBatch K0 n n]);
        elseif method == 2,
            recon = phantomdata;
        end

        recon = bsxfun(@plus, recon, model_vbias);
        
        if (params.sparseness > 0),
            recon = recon + model.sigma * randn(size(recon));
        end
        
        %% mean field hidden update
        
        neghidacts = reshape(sum(convs(recon, model_W, useCuda),2), [param_szBatch K nh nh]);
        neghidprobs = poolHidden(neghidacts, model_hbias, p, useCuda);
            
        if (params.verbose > 1),
            fprintf('.');
            errsum = errsum + sum(sum(sum(sum((batchdata - recon).^2))));
            if (params.verbose > 4),
                figure(2);imagesc(reshape(batchdata(1,1,:,:), [n n]));colormap gray;
                figure(1);imagesc(reshape(recon(1,1,:,:), [n n]));colormap gray;
                drawnow;
            end
        end
        
        %% contrast divergence update on params
        
        if (params.sparseness > 0),
%             hidact = hidact + reshape(sum(sum(sum(sigmoid(bsxfun(@plus, poshidacts, model_hbias)), 4), 3), 1), [1 K]);
            hidact = hidact + reshape(sum(sum(sum(pospoolprobs, 4), 3), 1), [1 K]);
        else
            dhbias = phbias * dhbias + ...
                reshape((sum(sum(sum(poshidprobs, 4), 3), 1) - sum(sum(sum(neghidprobs, 4), 3), 1))...
                / nh / nh / param_szBatch, [1 K]);
        end
        
        dvbias = pvbias * dvbias + ...
            reshape((sum(sum(sum(batchdata, 4), 3), 1) - sum(sum(sum(recon, 4), 3), 1))...
            / n / n / param_szBatch, [1 K0]);
        ddw = convs4(batchdata(:,:,m:n-m+1,m:n-m+1), poshidprobs(:,:,m:nh-m+1,m:nh-m+1), useCuda) ...
            - convs4(recon(:,:,m:n-m+1,m:n-m+1), neghidprobs(:,:,m:nh-m+1,m:nh-m+1), useCuda);
        dW = pW * dW + ddw / (nh - 2 * m + 2) / (nh - 2 * m + 2) / param_szBatch;
        
        model.vbias = model.vbias + params.epsvbias * dvbias;
        if params.sparseness <= 0,
            model.hbias = model.hbias + params.epshbias * dhbias; 
        end
        model.W = model.W + params.epsW * (dW  - params.decayw * model.W);
        
%         save dbgInfo model poshidacts poshidprobs poshidstates recon neghidacts neghidprobs model_W
%         if any(isnan(model.W(:))) || any(isnan(poshidacts(:))) || any(isnan(poshidprobs(:))) || any(isnan(poshidstates(:))) ...
%                 || any(isnan(recon(:))) || any(isnan(neghidacts(:))) || any(isnan(neghidprobs(:))),
%             return;
%         end
        if method == 2,
            phantom(batch_idx((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch),:,:,:) = reshape(sum(conve(neghidprobs, model_W, useCuda), 3), [param_szBatch K0 n n]);
        end
    end
    
    if params.sparseness > 0,
        hidact = hidact / np / np / N;
        hidq = hidq * lambdaq + hidact * (1 - lambdaq);
        dhbias = phbias * dhbias + ((params.sparseness) - (hidq));
        model.hbias = model.hbias + params.epshbias * dhbias;
        if params.verbose > 0,
            if (params.verbose > 1),
                fprintf('\n\terror:%f', errsum);
                if (params.sparseness > 0),
                    fprintf('\tsigma:%f', model.sigma);
                end
            end
            fprintf('\n\tsparseness: %f\thidbias: %f\n', sum(hidact) / K, sum(model.hbias) / K);
        end
%         if (model.sigma > 0.05),
%             model.sigma = model.sigma * 0.99;
%         end
    end
    
    if ~rem(iter, params.saveInterv),
        if (params.verbose > 3),
            figure(2);imagesc(batchdata(:,:,1));colormap gray;
            figure(1);imagesc(recon(:,:,1));colormap gray;
            drawnow;
        end
        if output_enabled,
            model.iter = iter;
            save(params.saveName, 'model', 'output', 'iter');
            if params.verbose > 1,  
                fprintf('Model and output saved at iteration %d\n', iter);
            end
        else 
            model.iter = iter;
            save(params.saveName, 'model', 'iter');
            if params.verbose > 1,
                fprintf('Model saved at iteration %d\n', iter);
            end
        end
    end
    
%     for i = 1:20,subplot(4,5,i);imagesc(reshape(whiten_data(model.W(1,i,:,:), uwhM),m,m));axis image off;end;colormap gray;drawnow;pause(0.1);
    for i = 1:16,subplot(4,4,i);imagesc(reshape(model.W(1,i,:,:),m,m));axis image off;end;colormap gray;drawnow;pause(0.1);
end
end