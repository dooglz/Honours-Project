#.rst:
# FindOpenCL
# ----------
#
# Try to find OpenCL
#
# Once done this will define::
#
#   OpenCL_FOUND          - True if OpenCL was found
#   OpenCL_INCLUDE_DIRS   - include directories for OpenCL
#   OpenCL_LIBRARIES      - link against this library to use OpenCL
#   OpenCL_VERSION_STRING - Highest supported OpenCL version (eg. 1.2)
#   OpenCL_VERSION_MAJOR  - The major version of the OpenCL implementation
#   OpenCL_VERSION_MINOR  - The minor version of the OpenCL implementation
#
# The module will also define two cache variables::
#
#   OpenCL_INCLUDE_DIR    - the OpenCL include directory
#   OpenCL_LIBRARY        - the path to the OpenCL library
#

#=============================================================================
# Copyright 2014 Matthaeus G. Chajdas
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

function(_FIND_OPENCL_VERSION inc_dir prefix)
  include(CheckCXXSymbolExists)
  include(CheckSymbolExists)
  include(CMakePushCheckState)
  set(CMAKE_REQUIRED_QUIET ${OpenCL_FIND_QUIETLY})
  CMAKE_PUSH_CHECK_STATE()
  foreach(VERSION "2_1" "2_0" "1_2" "1_1" "1_0")
    set(CMAKE_REQUIRED_INCLUDES "${inc_dir}")
    if(APPLE)
		check_cxx_symbol_exists(
			CL_VERSION_${VERSION}
			"${inc_dir}/OpenCL/cl.h"
			${prefix}OPENCL_VERSION_${VERSION}
		)
		if(!${prefix}OPENCL_VERSION_${VERSION})
			CHECK_SYMBOL_EXISTS(
				CL_VERSION_${VERSION}
				"${inc_dir}/OpenCL/cl.h"
				${prefix}OPENCL_VERSION_${VERSION}
			)
		endif()
    else()
		check_cxx_symbol_exists(
			CL_VERSION_${VERSION}
			"${inc_dir}/CL/cl.h"
			${prefix}OPENCL_VERSION_${VERSION}
		)
		if(!${prefix}OPENCL_VERSION_${VERSION})
			CHECK_SYMBOL_EXISTS(
				CL_VERSION_${VERSION}
				"${inc_dir}/CL/cl.h"
				${prefix}OPENCL_VERSION_${VERSION}
			)
		endif()
    endif()
	
    if(${prefix}OPENCL_VERSION_${VERSION})
      string(REPLACE "_" "." VERSION "${VERSION}")
      set(${prefix}OpenCL_VERSION_STRING ${VERSION} PARENT_SCOPE)
      string(REGEX MATCHALL "[0-9]+" version_components "${VERSION}")
      list(GET version_components 0 major_version)
      list(GET version_components 1 minor_version)
      set(${prefix}OpenCL_VERSION_MAJOR ${major_version} PARENT_SCOPE)
      set(${prefix}OpenCL_VERSION_MINOR ${minor_version} PARENT_SCOPE)
      break()
    endif()
  endforeach()
  CMAKE_POP_CHECK_STATE()
endfunction()

if(DEFINED ENV{CUDA_PATH})
	MESSAGE( "Env CUDA_PATH = $ENV{CUDA_PATH}" )
endif()

if(DEFINED ENV{AMDAPPSDKROOT})
	MESSAGE( "Env AMDAPPSDKROOT = $ENV{AMDAPPSDKROOT}" )
endif()

#  Find Includes
find_path(OpenCL_INCLUDE_DIR
  NAMES
    CL/cl.h OpenCL/cl.h
  PATHS
    ENV "PROGRAMFILES(X86)"
    ENV AMDAPPSDKROOT
    ENV INTELOCLSDKROOT
    ENV NVSDKCOMPUTE_ROOT
    ENV CUDA_PATH
    ENV ATISTREAMSDKROOT
	"/usr/local/cuda"
	"/opt/AMDAPP"
  PATH_SUFFIXES
    include
    OpenCL/common/inc
    "AMD APP/include"
)
if(OpenCL_INCLUDE_DIR)
	# If found, see if any other sdks are about:
	find_path(AMD_OpenCL_INCLUDE_DIR
	  NAMES
		CL/cl.h OpenCL/cl.h
	  PATHS
		ENV AMDAPPSDKROOT
		ENV ATISTREAMSDKROOT
	  PATH_SUFFIXES
		include
	  NO_DEFAULT_PATH 
	)
	if(AMD_OpenCL_INCLUDE_DIR)
		#MESSAGE("AMD Headders Found at: ${AMD_OpenCL_INCLUDE_DIR}")
		_FIND_OPENCL_VERSION(${AMD_OpenCL_INCLUDE_DIR} "AMD_")
	endif()
	find_path(NVIDIA_OpenCL_INCLUDE_DIR
	  NAMES
		CL/cl.h OpenCL/cl.h
	  PATHS
		ENV NVSDKCOMPUTE_ROOT
		ENV CUDA_PATH
		"/usr/local/cuda"
	  PATH_SUFFIXES
		include
	  NO_DEFAULT_PATH 
	)
	if(NVIDIA_OpenCL_INCLUDE_DIR)
		#MESSAGE("Nvidia Headders Found at: ${NVIDIA_OpenCL_INCLUDE_DIR}")
		_FIND_OPENCL_VERSION(${NVIDIA_OpenCL_INCLUDE_DIR} "NVIDIA_")
	endif()
endif()

_FIND_OPENCL_VERSION(${OpenCL_INCLUDE_DIR} "")

if(WIN32)
	if(CMAKE_SIZEOF_VOID_P EQUAL 4)
		#  32bit windows
		find_library(OpenCL_LIBRARY
			NAMES OpenCL
			PATHS
				ENV "PROGRAMFILES(X86)"
				ENV AMDAPPSDKROOT
				ENV INTELOCLSDKROOT
				ENV CUDA_PATH
				ENV NVSDKCOMPUTE_ROOT
				ENV ATISTREAMSDKROOT
			PATH_SUFFIXES
				"AMD APP/lib/x86"
				lib/x86
				lib/Win32
				OpenCL/common/lib/Win32
		)
		if(AMD_OpenCL_INCLUDE_DIR)
			find_library(AMD_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					ENV AMDAPPSDKROOT
					ENV ATISTREAMSDKROOT
				PATH_SUFFIXES
					"AMD APP/lib/x86"
					lib/x86
					lib/Win32
				NO_DEFAULT_PATH
			)
			if(AMD_OpenCL_LIBRARY)
				set(AMD_OpenCL_FOUND true)
			endif()
		endif()
		if(NVIDIA_OpenCL_INCLUDE_DIR)
			find_library(NVIDIA_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS 
					ENV CUDA_PATH
					ENV NVSDKCOMPUTE_ROOT
				PATH_SUFFIXES
					lib/x86
					lib/Win32
				NO_DEFAULT_PATH
			)
			if(NVIDIA_OpenCL_LIBRARY)
				set(NVIDIA_OpenCL_FOUND true)
			endif()
		endif()
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
		#  64bit windows
		find_library(OpenCL_LIBRARY
			NAMES OpenCL
			PATHS
				ENV "PROGRAMFILES(X86)"
				ENV AMDAPPSDKROOT
				ENV INTELOCLSDKROOT
				ENV CUDA_PATH
				ENV NVSDKCOMPUTE_ROOT
				ENV ATISTREAMSDKROOT
			PATH_SUFFIXES
				"AMD APP/lib/x86_64"
				lib/x86_64
				lib/x64
				OpenCL/common/lib/x64
		)
		if(AMD_OpenCL_INCLUDE_DIR)
			find_library(AMD_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					ENV AMDAPPSDKROOT
					ENV ATISTREAMSDKROOT
				PATH_SUFFIXES
					lib/x86_64
					lib/x64
				NO_DEFAULT_PATH
			)
			if(AMD_OpenCL_LIBRARY)
				set(AMD_OpenCL_FOUND true)
			endif()
		endif()
		if(NVIDIA_OpenCL_INCLUDE_DIR)
			find_library(NVIDIA_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS 
					ENV CUDA_PATH
					ENV NVSDKCOMPUTE_ROOT
				PATH_SUFFIXES
					lib/x86_64
					lib/x64
				NO_DEFAULT_PATH
			)
			if(NVIDIA_OpenCL_LIBRARY)
				set(NVIDIA_OpenCL_FOUND true)
			endif()
		endif()
  endif()
else() # Else linux
	if(CMAKE_SIZEOF_VOID_P EQUAL 4)
		# 32bit linux
		find_library(OpenCL_LIBRARY
			NAMES 
				OpenCL
			PATHS
				ENV LD_LIBRARY_PATH
				ENV OpenCL_LIBPATH
				ENV AMDAPPSDKROOT
				ENV AMDAPPSDKROOT/lib/x86_64
				/usr
			PATH_SUFFIXES
				nvidia
				lib
				lib/x86
		)
		if(AMD_OpenCL_INCLUDE_DIR)
			find_library(AMD_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					ENV AMDAPPSDKROOT
				PATH_SUFFIXES
					lib
					lib/x86
				NO_DEFAULT_PATH
			)
			if(AMD_OpenCL_LIBRARY)
				set(AMD_OpenCL_FOUND true)
			endif()
		endif()
		if(NVIDIA_OpenCL_INCLUDE_DIR)
			find_library(NVIDIA_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					/usr
				PATH_SUFFIXES
					nvidia
				NO_DEFAULT_PATH
			)
			if(NVIDIA_OpenCL_LIBRARY)
				set(NVIDIA_OpenCL_FOUND true)
			endif()
		endif()
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
		# 64bit linux
		find_library(OpenCL_LIBRARY
			NAMES 
				OpenCL
			PATHS
				ENV LD_LIBRARY_PATH
				ENV OpenCL_LIBPATH
				ENV AMDAPPSDKROOT
				ENV AMDAPPSDKROOT/lib/x86_64
				/usr
			PATH_SUFFIXES
				nvidia
				lib
				lib/x86_64
		)
		if(AMD_OpenCL_INCLUDE_DIR)
			find_library(AMD_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					ENV AMDAPPSDKROOT
				PATH_SUFFIXES
					lib
					lib/x86_64
				NO_DEFAULT_PATH
			)
			if(AMD_OpenCL_LIBRARY)
				set(AMD_OpenCL_FOUND true)
			endif()
		endif()
		if(NVIDIA_OpenCL_INCLUDE_DIR)
			find_library(NVIDIA_OpenCL_LIBRARY
				NAMES OpenCL
				PATHS
					/usr
				PATH_SUFFIXES
					nvidia
				NO_DEFAULT_PATH
			)
			if(NVIDIA_OpenCL_LIBRARY)
				set(NVIDIA_OpenCL_FOUND true)
			endif()
		endif()
	endif()
endif()



set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
#  set OpenCL_FOUND if all of REQUIRED_VARS == true
find_package_handle_standard_args(
  OpenCL
  FOUND_VAR OpenCL_FOUND
  REQUIRED_VARS OpenCL_LIBRARY OpenCL_INCLUDE_DIR
  VERSION_VAR OpenCL_VERSION_STRING)

mark_as_advanced(
  OpenCL_INCLUDE_DIR
  OpenCL_LIBRARY)
