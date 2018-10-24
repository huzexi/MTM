cmake_minimum_required(VERSION 2.8)
project(MTS)

# add OpenCV
set(OPENCV_DIR_HINT "/home/zeke/Libs/opencv/opencv-3.4.3/build")
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})

FIND_PACKAGE(OpenMP REQUIRED)

set(mts_src
	"${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ResultMonitor.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TrajectoriesRefiner.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/TaskConfig.h"
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tracker/KCF")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tracker/Struck")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tracker/CT")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tracker/DSST")

include_directories(${Struck_INCLUDE_DIRS})

add_executable(main ${mts_src})
target_link_libraries(main ${OpenCV_LIBS} )
target_link_libraries(main Struck_lib)
target_link_libraries(main kcf_lib)
target_link_libraries(main ct_lib)
target_link_libraries(main DSST_lib)
set_target_properties(main PROPERTIES
		CXX_STANDARD 11
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

