#DeployTest

# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h)

add_executable(DeployTest ${SRCS})

if(${OpenCL_FOUND})
	target_link_libraries (DeployTest ${OpenCL_LIBRARIES})
else(${OpenCL_FOUND}) 
	MESSAGE( "no opencl lib" )
endif(${OpenCL_FOUND})