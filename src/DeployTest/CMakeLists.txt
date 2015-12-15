#DeployTest

# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h *.cl *.cu)

FILE( GLOB all_exps exp_*)
source_group(experiments FILES ${all_exps})

#add_executable(DeployTest ${SRCS})
CUDA_ADD_EXECUTABLE(DeployTest ${SRCS})

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

if(UNIX)
	add_custom_command(
			TARGET DeployTest POST_BUILD
			COMMAND cp -f ${CMAKE_CURRENT_SOURCE_DIR}/*.cl $<TARGET_FILE_DIR:DeployTest>
			COMMENT "Copying ${srcdir_win} ${outdir_win}"
	)
else(WIN32) 
	FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR} srcdir_win) 
	FILE(TO_NATIVE_PATH $<TARGET_FILE_DIR:DeployTest> outdir_win) 
	add_custom_command(
		TARGET DeployTest POST_BUILD
		COMMAND xcopy /y ${srcdir_win}\\*.cl \"$<TARGET_FILE_DIR:DeployTest>\\\"
		COMMENT "XCopying ${srcdir_win} ${outdir_win}"
	)
endif()


#xcopy C:\Users\40082367\Desktop\Honours\src\DeployTest\*.cl C:\a



#MACRO(MAKE_PATH pathname)
#  # An extra \\ escape is necessary to get a \ through CMake's processing.
#  STRING(REPLACE "/" "\\" ${pathname} "${${pathname}}")
#  # Enclose with UNESCAPED quotes.  This means we need to escape our
#  # quotes once here, i.e. with \"
#  SET(${pathname} \"${${pathname}}\")
#ENDMACRO(MAKE_PATH)
#
#foreach(ff ${CL_PROGRAMS})
#  add_custom_command(	TARGET DeployTest POST_BUILD        # Adds a post-build event to MyTest
#	COMMAND ${CMAKE_COMMAND}
#	-E copy_if_different  # which executes "cmake - E copy_if_different..."
#	MAKE_PATH(${ff})    # <--this is in-file
#	$<TARGET_FILE_DIR:DeployTest>
#	COMMENT "Copying ${SRCFILE} ${TO_DIR}")
#	
#endforeach()

#add_custom_command(
#	TARGET DeployTest POST_BUILD        # Adds a post-build event to MyTest
#    COMMAND ${CMAKE_COMMAND}
#	-E copy_if_different  # which executes "cmake - E copy_if_different..."
#    ${CL_PROGRAMS}    # <--this is in-file
#    $<TARGET_FILE_DIR:DeployTest>
#	COMMENT "Copying ${SRCFILE} ${TO_DIR}"
#)