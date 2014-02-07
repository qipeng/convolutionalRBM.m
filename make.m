function make(option)

if (strcmpi(computer,'maci64')),
    setenv('PATH',[getenv('PATH'), ':', '/usr/local/cuda/bin']);
    setenv('DYLD_LIBRARY_PATH',['/usr/local/cuda/lib', ':', getenv('DYLD_LIBRARY_PATH')]);
end

if ~exist('option','var') || isempty(option),
    option = -1;
end

if option < 0,
    while 1,
        c = input('Do you want to setup your mex compiler first? (y / [n], Enter for No) ', 's');
        c = lower(c);
        if isempty(c) || c(1) == 'n',
            break;
        elseif c(1) == 'y',
            mex -setup
            break;
        end
    end
elseif bitand(option, 1),
    mex -setup
end

compileCuda = 0;

if option < 0,
    while 1,
        c = input('Do you want to compile the CUDA-MEX files? (y / [n], Enter for No) ', 's');
        c = lower(c);
        if isempty(c) || c(1) == 'n',
            compileCuda = 0;
            break;
        elseif c(1) == 'y',
            compileCuda = 1;
            break;
        end
    end
elseif bitand(option,2),
    compileCuda = 1;
end

fprintf('Compiling CPU-MEX files...\n');
mxlist = dir('mex/*.c');

for i = 1:length(mxlist),
    fprintf('(%d/%d) Compiling mex/%s...\n', i, length(mxlist), mxlist(i).name);
    try
        eval(sprintf('mex mex/%s', mxlist(i).name));
    catch exp,
        fprintf('[Error] Error compiling mex/%s, please refer to the error information for solution.\n', mxlist(i).name);
    end
end

if compileCuda,
    fprintf('Compiling CUDA-MEX files...\n');
    mxlist = dir('mex/*.cu');

    for i = 1:length(mxlist),
        fprintf('(%d/%d) Compiling mex/%s...\n', i, length(mxlist), mxlist(i).name);
        optionfile = 'nvmexopts.bat';
        if (strcmpi(computer, 'maci64')),
            optionfile = 'nvmexopts_maci64.sh';
        end
        try
            nvmex(sprintf('mex/%s', mxlist(i).name), '-f', optionfile);
        catch exp,
            fprintf('[Error] Error compiling mex/%s, please refer to the error information for solution.\n', mxlist(i).name);
        end
    end
end

fprintf('MEX compilation completed.\n');

end