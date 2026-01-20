set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
rem %userprofile%\anaconda3\python TSNE-precomputed.py %1 %2
@echo off
rem Find the first Python executable
for /f "usebackq tokens=*" %%i in (`where python`) do (
    set PYTHON_EXE=%%i
    goto :found_python
)

:found_python

rem Check if the Python executable was found
if "%PYTHON_EXE%"=="" (
    echo Python not found in PATH
    exit /b 1
)

rem Run the Python script with the found Python executable
"%PYTHON_EXE%" TSNE-precomputed.py %1 %2
