# find all *cpp, *h in current dir and assign to var SRCS
file( GLOB SRCS *.cpp *.h)

add_executable(DeployTest ${SRCS})
