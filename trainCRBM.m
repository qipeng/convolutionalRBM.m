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
%   Version: 0.2 alpha

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
param_mfIter = params.mfIter;
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
    if (~isfield(model,'W')), model.W = 0.1 * randn(m, m, K, K0);end
    if (~isfield(model,'vbias')), model.vbias = zeros(1, K0);end
    if (~isfield(model,'hbias')), model.hbias = ones(1, K) * hinit;end
else
    model.W = 0.01 * randn(m, m, K, K0);
    model.vbias = zeros(1, K0);
    model.hbias = ones(1, K) * hinit;
end

dW = 0;
dvbias = 0;
dhbias = 0;

pW = params.pW;
pvbias = params.pvbias;
phbias = params.phbias;

if output_enabled,
    output.x = zeros(np, np, K, N);
end

total_batches = floor(N / param_szBatch);

if params.verbose > 0,
    fprintf('Completed.\n');
end

hidq = params.sparseness;
lambdaq = 0.9;
if (params.sparseness > 0)
    sigma = 0.4;
else
    sigma = 1;    
end

for iter = 1:param_iter,
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
        batchdata = data.x(:,:,:,batch_idx((batch - 1) * param_szBatch + 1 : ...
            batch * param_szBatch));
        recon = batchdata;
%         pospoolprobs = zeros(np, np, K, param_szBatch);
        poshidprobs = zeros(nh, nh, K, param_szBatch);
%         negpoolprobs = zeros(np, np, K, param_szBatch);
        neghidprobs = zeros(nh, nh, K, param_szBatch);        
        
        %% positive phase

        %% mean field hidden update
        
        if output_enabled,
            output_x = zeros(np, np, K, param_szBatch);
        end
        model_W = model.W;
        model_hbias = model.hbias;
        model_vbias = model.vbias;
        
        for d = 1:param_szBatch,
            datum = recon(:, :, :, d);
            if output_enabled && iter == param_iter,
                pprobs = zeros(np, np, K);
            end
            
%             hres = convs(datum, model_W, useCuda);
%             for k0 = 1:K0,
                for k = 1:K,
                    hres = convs(datum, reshape(model_W(:, :, k, :),[m m K0]), useCuda);
                    if (any(any(any(any(isnan(hres)))))),
                        save nanInput datum model_W;
                        error('NaN after convs 1');
                    end
                    
%                     hprob = sigmoid(hres + model_hbias(k));

                    if output_enabled && iter == param_iter,
                        [hprob, pprob] = meanFieldInference((hres + model_hbias(k))/(sigma^2), p, 1);
                        pprobs(:,:,k) = pprobs(:,:,k) + pprob;
                    else
                        hprob = meanFieldInference((hres + model_hbias(k))/(sigma^2), p, 1);
                    end
                    
                    if (any(any(isnan(hprob)))),
                        save nanInput hres k k0 model_hbias p param_mfIter;
                        error('NaN after mean field 1');
                    end
                    
                    poshidprobs(:, :, k, d) = ...poshidprobs(:, :, k, d)...
                        + hprob;
                end
%             end
            
            if output_enabled && iter == param_iter,
                output_x(:,:,:,d) = pprobs / K0;
            end
        end
        
        if output_enabled && iter == param_iter,
            output.x(:,:,:,(batch - 1) * param_szBatch + 1:batch * param_szBatch) = output_x;
        end

        poshidprobs = poshidprobs / K0;
%         poshidstates = double(poshidprobs > rand(size(poshidprobs)));
%         poshidstates = poshidprobs;
%         pospoolprobs = pospoolprobs ./ K0;
        poshidstates = zeros(size(poshidprobs));
        for k = 1:K,
                for px = 1:np,
                    for py = 1:np,
                        rnd = rand;
                        acc = 0;
                        for h = 1:p*p,
%                             for hy = 1:p,
                            hx = floor((h-1)/p) + 1;
                            hy = mod((h-1),p) + 1;
                                acc = acc + poshidprobs((px-1)*p+hx,(py-1)*p+hy,k);
                                if (acc >= rnd)
                                    poshidstates((px-1)*p+hx,(py-1)*p+hy,k) = 1;
                                    break;
                                end
%                             end
                        end
                    end
                end
        end

        %% negative phase
        
        %% reconstruct data from hidden variables

        recon = zeros(n, n, K0, param_szBatch);

        for d = 1:param_szBatch,
            hidden = poshidstates(:, :, :, d);
%             vres = convemex(hidden, model_W);
            recon(:,:,:,d) = conve(hidden, model_W, useCuda);
%             for k = 1:K,
%                 hiddenk = hidden(:, :, k);
%                 for k0 = 1:K0,
%                     vres = conve(hiddenk, model_W(:, :, k, k0), useCuda);
%                     recon(:, :, k0, d) = recon(:, :, k0, d) + ...
%                         vres;
%                 end
%             end
             
%             for k0 = 1:K0,
%                 recon(:,:,k0,d) = sigmoid(recon(:,:,k0,d)+model_vbias(k0));
%             end
        end
        
        if (params.sparseness > 0),
            recon = recon + sigma * randn(size(recon));
        end

        %recon = recon ./ K;
        
        %% mean field hidden update
        
        for d = 1:param_szBatch,
            datum = recon(:, :, :, d);
%             hres = convs(datum, model_W(:,:,), useCuda);
            if (any(any(any(any(isnan(hres)))))),
                save nanInput datum model_W;
                error('NaN after convs 2');
            end
%             for k0 = 1:K0,
%                 datumk0 = datum(:,:,k0);
                for k = 1:K,
                    hres = convs(datum, reshape(model_W(:, :, k, :), [m m K0]), useCuda);
                    neghidprobs(:, :, k, d) =... neghidprobs(:, :, k, d)...
                        + meanFieldInference((hres + model_hbias(k)) / (sigma^2), p, 1);
                        ...+ sigmoid(hres + model_hbias(k));
                    
                    if (any(any(isnan(neghidprobs(:,:,k,d))))),
                        save nanInput hres k model_hbias p param_mfIter;
                        error('NaN after mean field 2');
                    end
                end
%             end
        end

        neghidprobs = neghidprobs / K0;
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
                / nh / nh / param_szBatch, [1 K]);
        end

        dvbias = pvbias * dvbias + ...
            reshape((sum(sum(sum(batchdata, 4), 2), 1) - sum(sum(sum(recon, 4), 2), 1))...
            / n / n / param_szBatch, [1 K0]);
        ddw = zeros(size(model.W));
        for d = 1:param_szBatch,
            datad = batchdata(:,:,:,d);
%             pospd = poshidprobs(:,:,:,d);
            pospd = poshidstates(:,:,:,d);
            recd = recon(:,:,:,d);
            negpd = neghidprobs(:,:,:,d);
            posprod = convs(datad, pospd, useCuda, 1);
            negprod = convs(recd, negpd, useCuda, 1);
            ddw = ddw + posprod - negprod;
%             for k = 1:K,
%                 for k0 = 1:K0,
%                     ddw(:, :, k, k0) = ddw(:, :, k, k0) + ...
%                         convs(batchdata(:, :, k0, d), poshidprobs(:, :, k, d), useCuda) -...
%                         convs(recon(:, :, k0, d), neghidprobs(:, :, k, d), useCuda);
%                 end
%             end
        end
        dW = pW * dW + ddw / nh / nh / param_szBatch;
        
        model.vbias = model.vbias + params.epsvbias * dvbias;
        if params.sparseness <= 0,
            model.hbias = model.hbias + params.epshbias * dhbias; 
        end
        model.W = model.W + params.epsW * (dW  - params.decayw * model.W);
    end
    
    if params.sparseness > 0,
        hidact = hidact / nh / nh / N * p * p;
        hidq = hidq * lambdaq + hidact * (1 - lambdaq);
        dhbias = phbias * dhbias + (params.sparseness - hidq);
        model.hbias = model.hbias + params.epshbias * dhbias;
        if params.verbose > 0,
            if (params.verbose > 1),
                fprintf('\n\terror:%f', errsum);
                if (params.sparseness > 0),
                    fprintf('\tsigma:%f', sigma);
                end
            end
            fprintf('\n\tsparseness: %f\thidbias: %f\n', sum(hidact) / K, sum(model.hbias) / K);
        end
        if (sigma > 0.05),
            sigma = sigma * 0.99;
        end
    end
    
    if ~rem(iter, params.saveInterv),
        if (params.verbose > 3),
            figure(2);imagesc(batchdata(:,:,1));colormap gray;
            figure(1);imagesc(recon(:,:,1));colormap gray;
            drawnow;
        end
        if output_enabled,
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
    
    for i = 1:10,subplot(2,5,i);imagesc(model.W(:,:,i));axis image off;end;colormap gray;drawnow;pause(0.1);
end
end