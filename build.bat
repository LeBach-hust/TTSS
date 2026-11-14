@echo off
setlocal
REM Build all four variants with MSYS2 GCC and MS-MPI.
REM Configure PATH so g++, mpicxx, and OpenMP runtime are available.

if not exist bin mkdir bin

echo Build script scaffold is ready.
echo Add compiler paths and build commands after the source implementations are added.
exit /b 0
