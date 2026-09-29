@echo off
setlocal
rem Double-click to start CloudClean. It opens in your browser at http://localhost:8765
rem The first start sets everything up (a few minutes; about 800 MB of disk); later starts take seconds.
rem Options go to "cloudclean serve", for example:  CloudClean.bat --host 0.0.0.0   (reachable from other PCs)
cd /d "%~dp0"
set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

if exist "%PY%" (
  rem set up again only when the requirements changed since last time (for example after "git pull")
  fc /b pyproject.toml "%VENV%\cloudclean-installed.toml" >nul 2>&1 && goto :run
  goto :install
)
echo Setting up CloudClean for the first time. This takes a few minutes...
call :make_venv || goto :error

:install
"%PY%" -m pip install --disable-pip-version-check -q --upgrade pip || goto :error
"%PY%" -m pip install --disable-pip-version-check -e ".[all]" || goto :error
copy /y pyproject.toml "%VENV%\cloudclean-installed.toml" >nul

:run
"%VENV%\Scripts\cloudclean.exe" serve %*
goto :eof

:make_venv
rem Open3D needs Python 3.10 - 3.12. Use one that is installed, otherwise let uv fetch 3.12 (no admin rights needed).
for %%v in (3.12 3.11 3.10) do (
  py -%%v -c "" >nul 2>&1 && py -%%v -m venv "%VENV%" && exit /b 0
)
if exist "%VENV%" rmdir /s /q "%VENV%"
where uv >nul 2>&1 || (
  echo Python 3.10 - 3.12 was not found. Installing uv, which fetches Python 3.12 for CloudClean...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || exit /b 1
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv venv --seed --python 3.12 "%VENV%" || exit /b 1
exit /b 0

:error
echo.
echo Setup failed (see the messages above). Check the internet connection and start CloudClean.bat again.
echo If it keeps failing, install Python 3.12 from https://www.python.org/downloads/ and try once more.
pause
exit /b 1
