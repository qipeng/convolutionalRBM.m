% Prepare example data
% Download one image from the van Hateren dataset
% URL: http://bethgelab.org/datasets/vanhateren/
if ~exist('data', 'file'),
    mkdir data
end

if ~exist('data/imk00001.imc', 'file'),
    fprintf('Downloading one image from the van Hateren dataset...');
    urlwrite('http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/imc/imk00001.imc',...
        'data/imk00001.imc');
    fprintf('Done.\n');
end

f1 = fopen('data/imk00001.imc', 'rb', 'ieee-be');
w = 1536; h = 1024;
% resize image and normalize pixels
data.x = imresize(fread(f1, [w, h], 'uint16'), [w h]/4 + 1);
data.x = double(data.x');
data.x = bsxfun(@rdivide, bsxfun(@minus, data.x, mean(data.x(:))), std(data.x(:)));
fclose(f1);

% Compile mex files
make(0);

params = getparams;
params.verbose = 4;
[model] = trainCRBM(data, params);