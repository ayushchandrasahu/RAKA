@echo off
echo ========================================================
echo  Deploying RAKA to Firebase Hosting
echo ========================================================
echo.
echo [1/3] Verifying Firebase CLI...
echo [2/3] Logging in to Google Firebase (browser will open)...
call npx firebase-tools login
echo.
echo [3/3] Deploying RAKA to Firebase Hosting...
call npx firebase-tools deploy --only hosting
echo.
echo ========================================================
echo  Deployment complete!
echo ========================================================
pause
