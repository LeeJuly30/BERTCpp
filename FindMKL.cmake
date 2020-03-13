# - Find Intel MKL
# Find the MKL libraries
#
# Options:
#
#   MKL_STATIC        :   use static linking
#   MKL_MULTI_THREADED:   use multi-threading
#   MKL_GNU_THREAD    :   use GNU OpenMP
#   MKL_SDL           :   Single Dynamic Library interface
#   MKL_INT64         :   use 64-bit integers
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.


include(FindPackageHandleStandardArgs)

if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(MKL_ARCH "64")
    set(MKL_ARCH_DIR "intel64")
    message(STATUS "[FindMKL] Architecture is 64-bit.")
else()
    set(MKL_ARCH "32")
    set(MKL_ARCH_DIR "ia32")
    message(STATUS "[FindMKL] Architecture is 32-bit.")
endif()

message(STATUS "[FindMKL] Architecture directory: ${MKL_ARCH_DIR}")
message(STATUS "[FindMKL] MKL_ROOT=${MKL_ROOT}")

# Find include directory
find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)

# Find libraries

# MKL is composed of four layers: Interface, Threading, Computational and RTL

if(MKL_SDL)
    message(STATUS "[FindMKL] Using Single Dynamically Linked version.")
    find_library(MKL_LIBRARY mkl_rt
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH_DIR}/)
else()
    ######################### Interface layer #######################
    # -lmkl_intel_lp64
    # -lmkl_intel_ilp64
    # -lmkl_intel
    #
    #
    message(STATUS "[FindMKL] Configuring interface layer...")
    if(${MKL_ARCH} STREQUAL "64")
        if(MKL_INT64)
            message(STATUS "[FindMKL] Using 64-bit integers.")
            set(MKL_INTERFACE_LIBNAME mkl_intel_ilp64)
        else()
            message(STATUS "[FindMKL] Using 32-bit integers.")
            set(MKL_INTERFACE_LIBNAME mkl_intel_lp64)
        endif()
    else()
        if(MKL_INT64)
            message(FATAL_ERROR
                "[FindMKL] Cannot use 64-bit integers on a 32-bit architecture.")
            return()
        else()
            message(STATUS "[FindMKL] Using 32-bit integers.")
            set(MKL_INTERFACE_LIBNAME mkl_intel)
        endif()
    endif()

    find_library(MKL_INTERFACE_LIB ${MKL_INTERFACE_LIBNAME}
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH_DIR}/ REQUIRED)
    message(STATUS "[FindMKL] ${MKL_INTERFACE_LIB}")

    ######################## Threading layer ########################
    # -lmkl_gnu_thread -lgomp
    # -lmkl_intel_thread -liomp
    # -lmkl_sequential
    #
    #
    message(STATUS "[FindMKL] Configuring threading layer...")
    if(MKL_MULTI_THREADED)
        message(STATUS "[FindMKL] Using multi-threaded interface.")
        if(MKL_GNU_THREAD)
            message(STATUS "[FindMKL] Using GNU OpenMP.")
            set(MKL_THREADING_LIBNAME mkl_gnu_thread)
            find_library(MKL_OMP_LIB gomp)
            message(STATUS "[FindMKL] libgomp: ${MKL_OMP_LIB}")
        else()
            message(STATUS "[FindMKL] Using Intel OpenMP.")
            set(MKL_THREADING_LIBNAME mkl_intel_thread)
            find_library(MKL_OMP_LIB iomp5
                PATHS ${INTEL_ROOT}/compiler/lib/${MKL_ARCH_DIR}/)
            message(STATUS "[FindMKL] libiomp: ${MKL_OMP_LIB}")
        endif()
    else()
        message(STATUS "[FindMKL] Using sequential interface.")
        set(MKL_THREADING_LIBNAME mkl_sequential)
    endif()

    find_library(MKL_THREADING_LIB ${MKL_THREADING_LIBNAME}
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH_DIR}/)

    ####################### Computational layer #####################
    # -lmkl_core
    #
    #
    message(STATUS "[FindMKL] Configuring computational layer...")
    find_library(MKL_CORE_LIB mkl_core
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH_DIR}/)

    set(MKL_LIBRARY
        ${MKL_INTERFACE_LIB}
        ${MKL_THREADING_LIB}
        ${MKL_CORE_LIB})
endif()

if(MKL_SDL)
    find_package_handle_standard_args(MKL DEFAULT_MSG
        MKL_LIBRARY)
else()
    find_package_handle_standard_args(MKL DEFAULT_MSG
        MKL_INTERFACE_LIB
        MKL_THREADING_LIB
        MKL_CORE_LIB)
endif()

if(MKL_FOUND)
    set(MKL_INCLUDES ${MKL_INCLUDE_DIR})
    set(MKL_LIBS ${MKL_LIBRARY})
    message(STATUS "[FindMKL] MKL_INCLUDES: ${MKL_INCLUDES}")
    message(STATUS "[FindMKL] MKL_LIBS: ${MKL_LIBRARY}")
endif()
