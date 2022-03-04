@echo off

set opts=-FC -GR- -EHa- -nologo -Zi
set code=%cd%

if not exist ..\build (
	echo Creating Build Folder
	mkdir ..\build
)

pushd ..\build
cl %opts% %code%\main.c -Femain.exe
popd
