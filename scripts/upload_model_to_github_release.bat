@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

set MODEL_FILE=models\checkpoints\best_model.pth
if not exist "%MODEL_FILE%" (
    echo [ERROR] %MODEL_FILE% 이 없습니다. 먼저 학습을 완료하거나 해당 경로에 파일을 두세요.
    pause
    exit /b 1
)

where gh >nul 2>&1
if errorlevel 1 (
    echo [ERROR] GitHub CLI(gh)가 설치되어 있지 않습니다.
    echo 설치: https://cli.github.com/
    echo 설치 후 터미널에서 gh auth login 으로 로그인하세요.
    pause
    exit /b 1
)

echo 모델 파일: %MODEL_FILE%
echo GitHub Release를 만들고 모델을 업로드합니다. 태그는 v1.0.0 사용 (이미 있으면 실패할 수 있음).
echo.
gh release create v1.0.0 "%MODEL_FILE%" --title "Model v1.0.0" --notes "best_model.pth for deployment"

if errorlevel 1 (
    echo.
    echo 이미 v1.0.0 릴리스가 있으면, 기존 릴리스에 파일만 추가하려면:
    echo   gh release upload v1.0.0 %MODEL_FILE% --clobber
    echo.
    pause
    exit /b 1
)

echo.
echo 완료. 아래 주소에서 다운로드 URL을 확인하세요.
echo GitHub 레포 - Releases - v1.0.0 - best_model.pth 우클릭 - 링크 주소 복사
echo 그 URL을 Render 환경변수 MODEL_URL 에 넣으면 됩니다.
echo.
pause
