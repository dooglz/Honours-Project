echo on
mkdir tests
cd tests
REM PINGPONG
REM FOR /L %%G IN (2,1,32) DO ..\build\bin_86_release\DeployTest.exe -b 4 -d 0,1 -e %%G,0
FOR /L %%G IN (2,1,31) DO ..\build\bin_86_release\DeployTest.exe -b 3 -d 0,1 -e %%G,0
pause > nul