function [model, output] = trainCRBM(data, params, oldModel)
% TRAINCRBM  Trains a convolutional restricted Boltzmann machine 
%   with the specified parameters.
%
%   [model output] = TRAINCRBM(data, params, oldModel)
%
%   data should be a structure, containing:
%       data.x      The input images / pooling states of the previous layer
%                   of CRBM. This matrix is 4-D the first three dimensions
%                   define an image (coloum-stored with a color channel),
%                   and the last dimension indexes through the batch of
%                   images
%
%   Written by: Peng Qi, Sep 27, 2012
%   Last Updated: Jul 22, 2013
%   Version: 0.3 alpha

if params.verbose > 0,
    fprintf('Starting training CRBM with the following parameters:\n');
    disp(params);
    fprintf('Initializing parameters...');
end

useCuda = params.useCuda;

if isfield(params, 'method'),
    if strcmp(params.method, 'CD'),
        method = 1; % Contrastive Divergence
    elseif strcmp(params.method, 'PCD'),
        method = 2; % Persistent Contrastive Divergence
    end
else
    method = 1;     % use Contrastive Divergence as default
end

%% initialization
N = size(data.x, 4);
Nfilters = params.nmap;
Wfilter = params.szFilter;
p = params.szPool;
H = size(data.x, 1);
W = size(data.x, 2);
colors = size(data.x, 3);
Hhidden = H - Wfilter + 1;
Whidden = W - Wfilter + 1;
Hpool = floor(Hhidden / p);
Wpool = floor(Whidden / p);
param_iter = params.iter;
param_szBatch = params.szBatch;
output_enabled = nargout > 1;

%vmasNfilters = conve(ones(nh), ones(m), useCuda);

hinit = 0;

if params.sparseness > 0,
    hinit = 0;
end

if exist('oldModel','var') && ~isempty(oldModel),
    model = oldModel;
    if (~isfield(model,'W')), 
        model.W = 0.01 * randn(Wfilter, Wfilter, colors, Nfilters);
    else
        if (size(model.W) ~= [Wfilter Wfilter colors Nfilters]), error('Incompatible input model.'); end
    end
    if (~isfield(model,'vbias')), model.vbias = zeros(1, colors);end
    if (~isfield(model,'hbias')), model.hbias = ones(1, Nfilters) * hinit;end
    if (~isfield(model,'sigma')),
        if (params.sparseness > 0)
            model.sigma = 0.1;
        else
            model.sigma = 1;    
        end
    end
else
    model.W = 0.01 * randn(Wfilter, Wfilter, colors, Nfilters);
    model.vbias = zeros(1, colors);
    model.hbias = ones(1, Nfilters) * hinit;
    if (params.sparseness > 0)
        model.sigma = 0.1;
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
    output.x = zeros(Hpool, Wpool, Nfilters, N);
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
    try
        load(sprintf('whitM_%d', params.szFilter));
    catch e,
        if (params.verbose > 1), fprintf('\nComputing whitening matrix...');end
        compWhitMatrix(data.x, params.szFilter);
        load(sprintf('whitM_%d', params.szFilter));
        if (params.verbose > 1), fprintf('Completed.\n');end
    end
    if (params.verbose > 0), fprintf('Whitening data...'); end
    data.x = whiten_data(data.x, whM, useCuda);
    if (params.verbose > 0), fprintf('Completed.\n'); end
end

if method == 2,
    phantom = randn(H, W, colors, N);
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
    
    hidact = zeros(1, Nfilters);
    errsum = 0;
    
    if (iter > 5),
        params.pW = .9;
        params.pvbias = 0;
        params.phbias = 0;
    end
    
    for batch = 1:total_batches,
        batchdata = data.x(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
            batch * param_szBatch));
        if method == 2,
            phantomdata = phantom(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch));
        end
        recon = batchdata;
        
        %% positive phase

        %% mean field hidden update
        
        model_W = model.W;
        model_hbias = model.hbias;
        model_vbias = model.vbias;
        
        poshidacts = convs(recon, model_W, useCuda);

        [poshidprobs, pospoolprobs, poshidstates] = poolHidden(poshidacts / model.sigma, model_hbias / model.sigma, p, useCuda);
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output_x = pospoolprobs;
        end
        
        if output_enabled && ~rem(iter, params.saveInterv),
            output.x((batch - 1) * param_szBatch + 1:batch * param_szBatch,:,:,:) = output_x;
        end
        
        %% negative phase
        
        %% reconstruct data from hidden variables

        if method == 1,
            recon = conve(poshidstates, model_W, useCuda);
        elseif method == 2,
            recon = phantomdata;
        end
        
        recon = bsxfun(@plus, recon, reshape(model_vbias, [1 1 colors]));

        if (params.sparseness > 0),
            recon = recon + model.sigma * randn(size(recon));
        end
        
        %% mean field hidden update
        
        neghidacts = convs(recon, model_W, useCuda);
        neghidprobs = poolHidden(neghidacts / model.sigma, model_hbias / model.sigma, p, useCuda);
            
        if (params.verbose > 1),
            fprintf('.');
            err = batchdata - recon;
            errsum = errsum + sum(err(:).^2);
            if (params.verbose > 4),
                figure(1);
                for i = 1:16,subplot(4,4,i);imagesc(model.W(:,:,:,i));axis image off;end;colormap gray;drawnow;
                figure(2);imagesc(batchdata(:,:,:,1));colormap gray;
                figure(3);imagesc(recon(:,:,:,1));colormap gray;
                drawnow;
            end
        end
        
        %% contrast divergence update on params
        
        if (params.sparseness > 0),
%             hidact = hidact + reshape(sum(sum(sum(sigmoid(bsxfun(@plus, poshidacts, model_hbias)), 4), 3), 1), [1 Nfilters]);
            hidact = hidact + reshape(sum(sum(sum(pospoolprobs, 4), 2), 1), [1 Nfilters]);
        else
            dhbias = phbias * dhbias + ...
                reshape((sum(sum(sum(poshidprobs, 4), 2), 1) - sum(sum(sum(neghidprobs, 4), 2), 1))...
                / Whidden / Hhidden / param_szBatch, [1 Nfilters]);
        end
        
        dvbias = pvbias * dvbias + ...
            reshape((sum(sum(sum(batchdata, 4), 2), 1) - sum(sum(sum(recon, 4), 2), 1))...
            / H / W / param_szBatch, [1 colors]);
        ddw = convs4(batchdata(Wfilter:H-Wfilter+1,Wfilter:W-Wfilter+1,:,:), poshidprobs(Wfilter:Hhidden-Wfilter+1,Wfilter:Whidden-Wfilter+1,:,:), useCuda) ...
            - convs4(    recon(Wfilter:H-Wfilter+1,Wfilter:W-Wfilter+1,:,:), neghidprobs(Wfilter:Hhidden-Wfilter+1,Wfilter:Whidden-Wfilter+1,:,:), useCuda);
        dW = pW * dW + ddw / (Hhidden - 2 * Wfilter + 2) / (Whidden - 2 * Wfilter + 2) / param_szBatch;
        
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
            phantom(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
                batch * param_szBatch)) = conve(neghidprobs, model_W, useCuda);
        end
    end
    
    if (params.verbose > 1),
        fprintf('\n\terror:%f', errsum);
    end
    
    if params.sparseness > 0,
        hidact = hidact / Hpool / Wpool / N;
        hidq = hidq * lambdaq + hidact * (1 - lambdaq);
        dhbias = phbias * dhbias + ((params.sparseness) - (hidq));
        model.hbias = model.hbias + params.epshbias * dhbias;
        if params.verbose > 0,
            if (params.verbose > 1),
                fprintf('\tsigma:%f', model.sigma);
            end
            fprintf('\n\tsparseness: %f\thidbias: %f\n', sum(hidact) / Nfilters, sum(model.hbias) / Nfilters);
        end
        if (model.sigma > 0.01),
            model.sigma = model.sigma * 0.99;
        end
    end
    
    if ~rem(iter, params.saveInterv),
        if (params.verbose > 3),
            figure(1);
            for i = 1:16,subplot(4,4,i);imagesc(model.W(:,:,:,i));axis image off;end;colormap gray;drawnow;
            figure(2);imagesc(batchdata(:,:,1));colormap gray;
            figure(3);imagesc(recon(:,:,1));colormap gray;
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
    
end
end