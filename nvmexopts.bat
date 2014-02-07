@echo off
rem MSVC80OPTS.BAT
rem
rem    Compile and link options used for building MEX-files
rem    using the Microsoft Visual C++ compiler version 8.0
rem
rem StorageVersion: 1.0
rem C++keyFileName: MSVC80OPTS.BAT
rem C++keyName: Microsoft Visual C++ 2005
rem C++keyManufacturer: Microsoft
rem C++keyVersion: 8.0
rem C++keyLanguage: C++
rem
rem    $Revision: 1.1.10.6 $  $Date: 2007/11/07 17:44:06 $
rem    Copyright 1984-2007 The MathWorks, Inc.
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************

set MATLAB=C:\Program Files\MATLAB\R2013a\
set VS80COMNTOOLS=%VS80COMNTOOLS%
set VSINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 10.0
set VCINSTALLDIR=%VSINSTALLDIR%\VC
set CUDA_INC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include
set CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\x64
set PATH=%VCINSTALLDIR%\BIN\;%VCINSTALLDIR%\PlatformSDK\bin;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\SDK\v2.0\bin;%VSINSTALLDIR%\Common7\Tools;%VSINSTALLDIR%\Common7\Tools\bin;%VCINSTALLDIR%\VCPackages;%MATLAB_BIN%;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Bin;%PATH%
set INCLUDE=%VCINSTALLDIR%\ATLMFC\INCLUDE;%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\PlatformSDK\INCLUDE;%VSINSTALLDIR%\SDK\v2.0\include;%MATLAB%\extern\include;%NVSDKDIR%\C\common\inc;%CUDA_INC_PATH%;%INCLUDE%
set LIB=C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\x64\;%VCINSTALLDIR%\ATLMFC\LIB;%VCINSTALLDIR%\LIB\amd64;%VCINSTALLDIR%\PlatformSDK\lib;%VSINSTALLDIR%\SDK\v2.0\lib;%MATLAB%\extern\lib\win64;%NVSDKDIR%\C\common\lib\x64;%CUDA_LIB_PATH%;%LIB%
set MW_TARGET_ARCH=win64

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
rem set COMPILER=cl
set COMPILER=nvcc
set COMPFLAGS= -c -arch compute_20 -Xcompiler "/c /Zp8 /GR /W0 /EHs /D_CRT_SECURE_NO_DEPRECATE /D_SCL_SECURE_NO_DEPRECATE /D_SECURE_SCL=0 /DMATLAB_MEX_FILE /nologo /MD"
set OPTIMFLAGS=-Xcompiler "/O2 /Oy- /DNDEBUG"
set DEBUGFLAGS=-Xcompiler "/Zi /Fd"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb""
set NAME_OBJECT= 
rem set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win64\microsoft
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /MAP /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib /implib:%LIB_NAME%.x /MACHINE:X64 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib curand.lib 
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/DEBUG /PDB:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%OUTDIR%%MEX_NAME%.map"
set POSTLINK_CMDS1=del %LIB_NAME%.x
set POSTLINK_CMDS2=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%";2 -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
rem set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest" 
