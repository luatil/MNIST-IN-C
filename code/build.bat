@echo off

set opts=-FC -GR- -EHa- -nologo -Zi
set code=%cd%

REM Compile test.cpp
REM pushd ..\build
REM cl %opts% %code%\test.cpp -Ftest.exe
REM popd

pushd ..\build
cl %opts% %code%\main.c -Femain.exe
popd
