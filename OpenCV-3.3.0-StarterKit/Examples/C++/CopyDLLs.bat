@ECHO OFF
IF NOT EXIST Debug MKDIR Debug
FOR %%f IN (..\..\OpenCV\x86\vc15\bin\*330d.dll) DO IF NOT EXIST Debug\%%~nxf COPY %%f Debug >NUL
IF NOT EXIST Debug\opencv_ffmpeg330.dll COPY ..\..\OpenCV\x86\vc15\bin\opencv_ffmpeg330.dll Debug
IF NOT EXIST Release MKDIR Release
FOR %%f IN (..\..\OpenCV\x86\vc15\bin\*330.dll) DO IF NOT EXIST Release\%%~nxf COPY %%f Release >NUL