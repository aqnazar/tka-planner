@echo off
rem One click: start the local planner, open the browser, load the case, carve the plan.
rem Research and demonstration only. Not a medical device.

setlocal
cd /d "%~dp0"

set "PORT=8731"
set "CASES=%~dp0data\cases"
set "CASE=first"
set "SIDE=left"
set "COMMIT=1"

rem The implant library is a separate design output, not part of this repository.
rem Point TKA_IMPLANT_LIBRARY at your own folder of component meshes, or drop them
rem in data\implants. Without one the planner still measures, aligns and resects;
rem only the components are missing from the scene.
if defined TKA_IMPLANT_LIBRARY (set "LIBRARY=%TKA_IMPLANT_LIBRARY%") else (set "LIBRARY=%~dp0data\implants")

rem Python: prefer the launcher, fall back to whatever is on PATH.
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"

if not exist "%CASES%" (
  echo No case folder at "%CASES%".
  echo Put one folder per patient there, each holding FD1Left.stl and TD1Left.stl.
  echo.
  pause
  exit /b 1
)

rem Open the browser as soon as the server answers, not before.
set "URL=http://127.0.0.1:%PORT%/?case=%CASE%&side=%SIDE%&commit=%COMMIT%"
set "WAIT=for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2 -UseBasicParsing | Out-Null; Start-Process '%URL%'; exit } catch { Start-Sleep -Milliseconds 500 } }"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "%WAIT%"

echo Starting the TKA planner on port %PORT%.
echo   cases   %CASES%
if exist "%LIBRARY%" (echo   library %LIBRARY%) else (echo   library none - set TKA_IMPLANT_LIBRARY to seat implants on the plan)
echo The browser will open by itself. Close this window to stop the planner.
echo.

if exist "%LIBRARY%" (
  %PY% -m tka_planner.server --cases "%CASES%" --library "%LIBRARY%" --port %PORT% --no-browser
) else (
  %PY% -m tka_planner.server --cases "%CASES%" --port %PORT% --no-browser
)

endlocal
