#DeployTest

# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h)

add_executable(DeployTest ${SRCS})

if(${OpenCL_FOUND})
	if(UNIX)
	target_link_libraries (DeployTest ${OpenCL_LIBRARIES} pthread)
	else()
	target_link_libraries (DeployTest ${OpenCL_LIBRARIES})
	endif()
else(${OpenCL_FOUND}) 
	MESSAGE( "no opencl lib" )
	message( FATAL_ERROR "Could Not locate OpenCL Library, Exiting cmake" )
endif(${OpenCL_FOUND})