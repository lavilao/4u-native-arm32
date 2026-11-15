cmake_minimum_required(VERSION 3.14)

# mh4n: the app is built for arm64-v8a and, as its 32-bit build, armeabi-v7a (a 32-bit ARM
# Adreno phone can still load its own drivers; the bcenabler patch is AArch64-only data but
# only ever runs on an Adreno that understands it)
if(NOT ${CMAKE_ANDROID_ARCH_ABI} STREQUAL arm64-v8a AND NOT ${CMAKE_ANDROID_ARCH_ABI} STREQUAL armeabi-v7a)
        message(FATAL_ERROR "Unsupported target architecture: ${CMAKE_ANDROID_ARCH_ABI}. Please make an issue on the repo!")
endif()

project(adrenotools LANGUAGES CXX C)

set(GEN_INSTALL_TARGET OFF CACHE BOOL "")

add_subdirectory(lib/linkernsbypass)

set(LIB_SOURCES src/bcenabler.cpp
                src/driver.cpp
                include/adrenotools/bcenabler.h
                include/adrenotools/driver.h
                include/adrenotools/priv.h)

add_library(adrenotools ${LIB_SOURCES})

target_include_directories(adrenotools PUBLIC include)
target_include_directories(adrenotools PRIVATE .)
target_compile_options(adrenotools PRIVATE -Wall -Wextra)
target_link_libraries(adrenotools android linkernsbypass)

add_subdirectory(src/hook)
if (${BUILD_SHARED_LIBS})
    target_link_options(adrenotools PRIVATE "-Wl,-s,--exclude-libs,liblinkernsbypass.a")
endif()
if (${GEN_INSTALL_TARGET})
  configure_file(adrenotools.pc.in adrenotools.pc @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/adrenotools.pc DESTINATION lib/pkgconfig)
  install(TARGETS adrenotools main_hook file_redirect_hook gsl_alloc_hook hook_impl)
  install(DIRECTORY include/
          DESTINATION include
          FILES_MATCHING PATTERN "*.h*")
endif()
