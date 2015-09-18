#DeployTest

# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h *.cl)

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

file(GLOB CL_PROGRAMS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cl" 
)

add_custom_command(
	TARGET DeployTest POST_BUILD        # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND}
	-E copy_if_different  # which executes "cmake - E copy_if_different..."
    ${CL_PROGRAMS}    # <--this is in-file
    $<TARGET_FILE_DIR:DeployTest>
	COMMENT "Copying ${SRCFILE} ${TO_DIR}"
)