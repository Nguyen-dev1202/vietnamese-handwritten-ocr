@echo off
title Vietnamese Handwritten OCR - SIC AI (Team 3)
echo ======================================================================
echo    HE THONG NHAN DANG CHU VIET TAY TIENG VIET - SIC AI (TEAM 3)
echo ======================================================================
echo.
echo [1/2] Dang kich hoat moi truong Miniconda: ocr_sic ...
call D:\miniconda\condabin\activate.bat ocr_sic

echo [2/2] Dang khoi dong Web Studio app.py ...
echo Trinh duyet web se tu dong mo tai dia chi: http://127.0.0.1:7860
echo.
python app.py
pause

