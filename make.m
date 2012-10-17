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

compileCuda = 0;

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

fprintf('Compiling CPU-MEX files...\n');
mxlist = dir('mex/*.cpp');

for i = 1:length(mxlist),
    fprintf('(%d/%d) Compiling mex/%s...\n', i, length(mxlist), mxlist(i).name);
    eval(sprintf('mex mex/%s', mxlist(i).name));
end

if compileCuda,
    fprintf('Compiling CUDA-MEX files...\n');
    mxlist = dir('mex/*.cu');

    for i = 1:length(mxlist),
        fprintf('(%d/%d) Compiling mex/%s...\n', i, length(mxlist), mxlist(i).name);
        eval(sprintf('nvmex -f nvmexopts.bat mex/%s', mxlist(i).name));
    end
end

fprintf('MEX compilation completed.\n');