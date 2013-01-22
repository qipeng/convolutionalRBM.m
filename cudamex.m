function cudamex(file)
    arch = computer;
    if ispc(),
        if strcmp(arch, 'PCWIN'),
            eval(sprintf('nvmex -f nvmexopts.bat %s', file));
        else
            error('64-bit Windows not supported yet.');
        end
    elseif isunix(),
        system('./nvmexsetup.sh');
        [~,name,~] = fileparts(file);
        eval(sprintf('!nvcc -c %s -Xcompiler -fPIC -I ${MATLAB}/extern/include', file));
        eval(sprintf('mex %s.o -L ${CUDA}/lib64 -L ${CUDA}/lib -lcudart -lcurand', name));
    else
        error('Unidentified operating system.');
    end
end