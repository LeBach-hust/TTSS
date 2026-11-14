@echo off
setlocal EnableDelayedExpansion

:: Script bien dich 10 chuong trinh (4 che do: seq, omp, mpi, hybrid)
:: Cach dung:
::   build.bat              Bien dich tat ca 4 che do (tu bo qua file chua co)
::   build.bat seq          Chi bien dich che do tuan tu (2 file)
::   build.bat omp          Chi bien dich che do OpenMP (2 file)
::   build.bat mpi          Chi bien dich che do MPI (3 file)
::   build.bat hybrid       Chi bien dich che do Hybrid MPI+OMP (3 file)
::   build.bat clean        Xoa thu muc bin/

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

if /I "%MODE%"=="clean" (
    echo [CLEAN] Dang xoa thu muc bin/...
    if exist bin rmdir /s /q bin
    echo [DONE] Da xoa sach thu muc bin.
    exit /b 0
)

:: Kiem tra tham so hop le
if /I not "%MODE%"=="all" if /I not "%MODE%"=="seq" if /I not "%MODE%"=="omp" if /I not "%MODE%"=="mpi" if /I not "%MODE%"=="hybrid" (
    echo [ERROR] Che do khong hop le: "%MODE%"
    echo Cach dung: build.bat [all ^| seq ^| omp ^| mpi ^| hybrid ^| clean]
    exit /b 1
)

:: Tao thu muc bin/ chua file .exe
if not exist bin mkdir bin

:: Cau hinh trinh bien dich C++ (MSYS2 MinGW-w64 g++)
set "CXX=g++"
set "COMMON_FLAGS=-O3 -march=native -std=c++17 -Wall -Wextra -Iinclude"

:: Tu dong do tim MS-MPI SDK tren Windows (ho tro ca MSYS2 msmpi va Microsoft SDK)
set "MPI_INC="
set "MPI_LIB=-lmsmpi"

if defined MSMPI_INC (
    set "MPI_INC=-I"%MSMPI_INC%" -I"%MSMPI_INC%\x64""
) else if exist "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h" (
    set "MPI_INC=-I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64""
)

if defined MSMPI_LIB64 (
    set "MPI_LIB=-L"%MSMPI_LIB64%" -lmsmpi"
) else if exist "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib" (
    set "MPI_LIB=-L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi"
)

:: 4 che do co bien dich (Flags)
set "FLAGS_SEQ=%COMMON_FLAGS% -fopenmp-simd"
set "FLAGS_OMP=%COMMON_FLAGS% -fopenmp"
set "FLAGS_MPI=%COMMON_FLAGS% %MPI_INC%"
set "FLAGS_HYB=%COMMON_FLAGS% -fopenmp %MPI_INC%"

set /a BUILT_COUNT=0
set /a SKIP_COUNT=0
set /a FAIL_COUNT=0

echo ============================================================
echo  Bat dau bien dich (Che do: %MODE%)
echo ============================================================

:: --- 1. Che do SEQ (2 chuong trinh) ---
if /I "%MODE%"=="all" (set "RUN_SEQ=1") else if /I "%MODE%"=="seq" (set "RUN_SEQ=1") else (set "RUN_SEQ=0")
if "%RUN_SEQ%"=="1" (
    call :compile "seq" "src/dot/dot_seq.cpp"       "bin/dot_seq.exe"       "!FLAGS_SEQ!" ""
    call :compile "seq" "src/matmul/matmul_seq.cpp" "bin/matmul_seq.exe"    "!FLAGS_SEQ!" ""
)

:: --- 2. Che do OMP (2 chuong trinh) ---
if /I "%MODE%"=="all" (set "RUN_OMP=1") else if /I "%MODE%"=="omp" (set "RUN_OMP=1") else (set "RUN_OMP=0")
if "%RUN_OMP%"=="1" (
    call :compile "omp" "src/dot/dot_omp.cpp"       "bin/dot_omp.exe"       "!FLAGS_OMP!" ""
    call :compile "omp" "src/matmul/matmul_omp.cpp" "bin/matmul_omp.exe"    "!FLAGS_OMP!" ""
)

:: --- 3. Che do MPI (3 chuong trinh) ---
if /I "%MODE%"=="all" (set "RUN_MPI=1") else if /I "%MODE%"=="mpi" (set "RUN_MPI=1") else (set "RUN_MPI=0")
if "%RUN_MPI%"=="1" (
    call :compile "mpi" "src/dot/dot_mpi.cpp"          "bin/dot_mpi.exe"          "!FLAGS_MPI!" "!MPI_LIB!"
    call :compile "mpi" "src/matmul/cannon_mpi.cpp"    "bin/cannon_mpi.exe"       "!FLAGS_MPI!" "!MPI_LIB!"
    call :compile "mpi" "src/matmul/cannon_mpi_nb.cpp" "bin/cannon_mpi_nb.exe"    "!FLAGS_MPI!" "!MPI_LIB!"
)

:: --- 4. Che do HYBRID (3 chuong trinh) ---
if /I "%MODE%"=="all" (set "RUN_HYB=1") else if /I "%MODE%"=="hybrid" (set "RUN_HYB=1") else (set "RUN_HYB=0")
if "%RUN_HYB%"=="1" (
    call :compile "hybrid" "src/dot/dot_hybrid.cpp"          "bin/dot_hybrid.exe"          "!FLAGS_HYB!" "!MPI_LIB!"
    call :compile "hybrid" "src/matmul/cannon_hybrid.cpp"    "bin/cannon_hybrid.exe"       "!FLAGS_HYB!" "!MPI_LIB!"
    call :compile "hybrid" "src/matmul/cannon_hybrid_nb.cpp" "bin/cannon_hybrid_nb.exe"    "!FLAGS_HYB!" "!MPI_LIB!"
)

echo ============================================================
echo  Ket qua: Thanh cong: !BUILT_COUNT! ^| Bo qua (chua co file): !SKIP_COUNT! ^| Loi: !FAIL_COUNT!
echo ============================================================

if !FAIL_COUNT! GTR 0 exit /b 1
exit /b 0

:: Ham con bien dich 1 chuong trinh (tu bo qua neu file .cpp chua ton tai)
:compile
set "TAG=%~1"
set "SRC=%~2"
set "OUT=%~3"
set "CXXFLAGS=%~4"
set "LDFLAGS=%~5"

if not exist "%SRC%" (
    echo [SKIP] [%TAG%] Chua co file %SRC%
    set /a SKIP_COUNT+=1
    goto :eof
)

echo [BUILD] [%TAG%] %SRC% -^> %OUT%
%CXX% %CXXFLAGS% "%SRC%" -o "%OUT%" %LDFLAGS%
if errorlevel 1 (
    echo [FAIL]  [%TAG%] Loi bien dich %SRC%
    set /a FAIL_COUNT+=1
) else (
    echo [OK]    [%TAG%] Da tao %OUT%
    set /a BUILT_COUNT+=1
)
goto :eof