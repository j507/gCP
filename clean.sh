#!/bin/bash
find . -name "CMakeCache.txt" -type f -delete
find . -name "MakeFile" -type f -delete
find . -name "cmake_install.cmake" -type f -delete
find . -name "CTestTestfile.cmake" -type f -delete
find . -name "CMakeFiles" -type d -exec rm -rf {} \;
cmake CMakeLists.txt
