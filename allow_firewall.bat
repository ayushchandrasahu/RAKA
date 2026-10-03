@echo off
echo ========================================================
echo  Configuring Windows Firewall for RAKA on Port 8000
echo ========================================================
netsh advfirewall firewall add rule name="RAKA-Port-8000" dir=in action=allow protocol=TCP localport=8000
powershell -Command "Set-NetConnectionProfile -InterfaceAlias Wi-Fi -NetworkCategory Private -ErrorAction SilentlyContinue"
echo.
echo [OK] Port 8000 is now ALLOWED for other devices on your network!
echo.
pause
