::# run this file as "cmd < sync_code_windows.sh update"


@echo off

REM Check if a commit message argument is provided
if "%~1"=="" (
    echo Usage: %0 ^<commit_message^>
    exit /b 1
)

REM Extract the commit message from the first argument
set commit_message=%~1

echo "argument"
echo %~1

REM Add all changes, commit with the provided message, and push
git add -A
git commit -m "%commit_message%"
git push