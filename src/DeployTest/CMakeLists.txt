#DeployTest

# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h)

add_executable(DeployTest ${SRCS})

if(${OpenCL_FOUND})
	target_link_libraries (DeployTest ${OpenCL_LIBRARIES} pthread)
else(${OpenCL_FOUND}) 
	MESSAGE( "no opencl lib" )
	message( FATAL_ERROR "Could Not locate OpenCL Library, Exiting cmake" )
endif(${OpenCL_FOUND})