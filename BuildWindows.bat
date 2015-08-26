echo off
where /q cmake
IF %ERRORLEVEL%==1 (
    Echo cmake not in path
	SET p86=C:\Program Files (x86)
	PATH=%path%;%ProgramFiles(x86)%\CMake\bin
	PATH=%path%;%ProgramFiles%\\CMake\bin
 )
where /q cmake
IF %ERRORLEVEL%==1 (
    Echo Couldn't find cmake.exe
) ELSE ( 
   mkdir build
   cd build
   echo Cmake building for VS2013 x86
   cmake -G "Visual Studio 12 2013" ../
   echo cmake complete, open the sln in the build dir
 )
pause > nul