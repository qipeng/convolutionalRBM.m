function [model output] = trainCRBM(data, params, oldModel)
% TRAINCRBM  Trains a convolutional restricted Boltzmann machine 
%   with the specified parameters.
%
%   [model output] = TRAINCRBM(data, params, oldModel)
%
%   data should be a structure, containing:
%       data.x      The input images / pooling states of the previous layer
%                   of CRBM. This matrix is 4-D, where the first three
%                   dimensions define an image, and the fourth indexes
%                   through the images
%
%   See Also MEXTRAINCRBM
%
%   Written by: Peng Qi, Sep 27, 2012
%   Version: 0.1 alpha

if params.verbose > 0,
    fprintf('Starting training CRBM with the following parameters:\n');
    disp(params);
    fprintf('Initializing parameters...');
end

useCuda = params.useCuda;

if useCuda,
    meanFieldInference = @mfInfer_cuda;
else
    meanFieldInference = @mfInfer;
end

%% initialization
N = size(data.x, 4);
K = params.nmap;
m = params.szFilter;
p = params.szPool;
n = size(data.x, 1);
K0 = size(data.x, 3);
nh = n - m + 1;
np = floor(nh / p);

vmask = convd(ones(nh), ones(m), useCuda);

if exist('oldModel','var') && ~isempty(oldModel),
    model = oldModel;
else
    model.W = 0.1 * randn(m, m, K, K0);
    model.vbias = zeros(1, K0);
    model.hbias = ones(1, K) * (-1);
end

dW = 0;
dvbias = 0;
dhbias = 0;

pW = params.pW;
pvbias = params.pvbias;
phbias = params.phbias;

if nargout > 1,
    output.x = zeros(np, np, K, N);
    output.dim = np;
    output.depth = K;
end

total_batches = floor(N / params.szBatch);

if params.verbose > 0,
    fprintf('Completed.\n');
end

for iter = 1:params.iter,
    % shuffule data
    batch_idx = randperm(N);
    
    if params.verbose > 0,
        fprintf('Iteration %d\n', iter);
        if params.verbose > 1,
            fprintf('Batch progress (%d total): ', total_batches);
        end
    end
    
    hidact = zeros(1, K);
    errsum = 0;
    
    for batch = 1:total_batches,
        batchdata = data.x(:,:,:,batch_idx((batch - 1) * params.szBatch + 1 : ...
            batch * params.szBatch));
        recon = batchdata;
%         pospoolprobs = zeros(np, np, K , params.szBatch);
        poshidprobs = zeros(nh, nh, K , params.szBatch);
%         negpoolprobs = zeros(np, np, K , params.szBatch);
        neghidprobs = zeros(nh, nh, K, params.szBatch);
        
        
        
        %% positive phase

        %% mean field hidden update
        for d = 1:params.szBatch,
            datum = recon(:, :, :, d);
            if nargout > 1 && iter == params.iter,
                pprobs = zeros(np, np, K);
            end
            
            for k0 = 1:K0,
                datumk0 = datum(:,:,k0);
                for k = 1:K,
                    hres = convu(datumk0, model.W(:, :, k, k0), useCuda);
                    
                    if nargout > 1 && iter == params.iter,
                        [hprob, pprob] = meanFieldInference(hres + model.hbias(k), p, params.mfIter);
                        pprobs(:,:,k) = pprobs(:,:,k) + pprob;
                    else
                        hprob = meanFieldInference(hres + model.hbias(k), p, params.mfIter);
                        if (any(any(any(isnan(hprob))))), save nanInput hres model p params; error('hprob NaN'); end
                    end
                    
                    poshidprobs(:, :, k, d) = poshidprobs(:, :, k, d)...
                        + hprob;
                end
            end
            
            if nargout > 1 && iter == params.iter,
                output.x(:, :, :, (batch - 1) * params.szBatch + d) = pprobs / K0;
            end
        end

        poshidprobs = poshidprobs ./ K0;
%         pospoolprobs = pospoolprobs ./ K0;

        %% negative phase
        
        %% reconstruct data from hidden variables

        recon = zeros(n, n, K0, params.szBatch);

        for d = 1:params.szBatch,
            hidden = poshidprobs(:, :, :, d);
            for k = 1:K,
                hiddenk = hidden(:, :, k);
                for k0 = 1:K0,
                    vres = convd(hiddenk, model.W(:, :, k, k0), useCuda)./vmask;
%                         recon(:, :, k0, d) = recon(:, :, k0, d) + ...
%                             1 ./ (1 + exp(-(vres + model.vbias(k0))));
                    recon(:, :, k0, d) = recon(:, :, k0, d) + ...
                        1./(1+exp(-(vres + model.vbias(k0))));
                end
            end
        end

        recon = recon ./ K;

        %% mean field hidden update
        for d = 1:params.szBatch,
            datum = recon(:, :, :, d);
            for k0 = 1:K0,
                datumk0 = datum(:,:,k0);
                for k = 1:K,
                    hres = convu(datumk0, model.W(:, :, k, k0), useCuda);
                    neghidprobs(:, :, k, d) = neghidprobs(:, :, k, d)...
                        + meanFieldInference(hres + model.hbias(k), p, params.mfIter);
                end
            end
        end

        neghidprobs = neghidprobs ./ K0;
%         negpoolprobs = negpoolprobs ./ K0;
            
        
        if (params.verbose > 1),
            fprintf('.');
            errsum = errsum + sum(sum(sum(sum((batchdata - recon).^2))));
            if (params.verbose > 4),
                figure(2);imagesc(batchdata(:,:,1));colormap gray;
                figure(1);imagesc(recon(:,:,1));colormap gray;
                drawnow;
            end
        end
        
        %% contrast divergence update on params
        
        if (params.sparseness > 0),
            hidact = hidact + reshape(sum(sum(sum(poshidprobs, 4), 2), 1), [1 K]);
        else
            dhbias = phbias * dhbias + ...
                reshape((sum(sum(sum(poshidprobs, 4), 2), 1) - sum(sum(sum(neghidprobs, 4), 2), 1))...
                / nh / nh / params.szBatch, [1 K]);
        end

        dvbias = pvbias * dvbias + ...
            reshape((sum(sum(sum(batchdata, 4), 2), 1) - sum(sum(sum(recon, 4), 2), 1))...
            / n / n / params.szBatch, [1 K0]);
        ddw = zeros(size(model.W));
        for d = 1:params.szBatch,
            for k = 1:K,
                for k0 = 1:K0,
                    ddw(:, :, k, k0) = ddw(:, :, k, k0) + ...
                        convu(batchdata(:, :, k0, d), poshidprobs(:, :, k, d), useCuda) -...
                        convu(recon(:, :, k0, d), neghidprobs(:, :, k, d), useCuda);
                end
            end
        end
        dW = pW * dW + ddw / params.szBatch / K0;
        
        model.vbias = model.vbias + params.epsvbias * dvbias;
        if params.sparseness <= 0,
            model.hbias = model.hbias + params.epshbias * dhbias; 
        end
        model.W = model.W + params.epsW * (dW  - params.decayw * model.W);
    end
    
    if params.sparseness > 0,
        hidact = hidact / nh / nh / N;
        model.hbias = model.hbias + params.epshbias * (params.sparseness - hidact);
        if params.verbose > 0,
            if (params.verbose > 1),
                fprintf('\n\terror:%f\n', errsum);
            end
            fprintf('\tsparseness: %f\thidbias: %f\n', sum(hidact) / K, sum(model.hbias) / K);
        end
    end
    
    if ~rem(iter, params.saveInterv),
        if nargout > 1,
            save(params.saveName, 'model', 'output', 'iter');
            if params.verbose > 1,
                fprintf('Model and output saved at iteration %d\n', iter);
            end
        else 
            save(params.saveName, 'model', 'iter');
            if params.verbose > 1,
                fprintf('Model saved at iteration %d\n', iter);
            end
        end
    end
end