cmake_minimum_required(VERSION 3.14)

# mh4n: the app is built for arm64-v8a and, as its 32-bit build, armeabi-v7a (the soname
# patcher rewrites the strings of whatever ELF it is handed, 32- or 64-bit, and the namespace
# bypass itself is the bionic linker's, either way)
if(NOT ${CMAKE_ANDROID_ARCH_ABI} STREQUAL arm64-v8a AND NOT ${CMAKE_ANDROID_ARCH_ABI} STREQUAL armeabi-v7a)
        message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_ANDROID_ARCH_ABI}. Please make an issue on the repo!")
endif()

project(linkernsbypass LANGUAGES CXX)

set(SOURCES android_linker_ns.cpp
            android_linker_ns.h
            elf_soname_patcher.cpp
            elf_soname_patcher.h)

add_library(linkernsbypass STATIC ${SOURCES})


target_compile_options(linkernsbypass PRIVATE -Wall -Wextra)
target_link_libraries(linkernsbypass android dl)
target_include_directories(linkernsbypass PUBLIC .)
