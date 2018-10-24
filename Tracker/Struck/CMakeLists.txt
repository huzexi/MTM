cmake_minimum_required(VERSION 2.6)
project(Struck)

find_package(OpenCV REQUIRED)

if(NOT DEFINED EIGEN3_INCLUDE_DIRS)
	find_package(Eigen3 QUIET)
	if(NOT Eigen3_FOUND)
		message(FATAL_ERROR "Eigen3 is not found.")
	endif()
endif()
include_directories(${EIGEN3_INCLUDE_DIRS})

set(Struck_src
	"${CMAKE_CURRENT_SOURCE_DIR}/StruckWrapper.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Config.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Features.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/HaarFeature.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/HaarFeatures.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/HistogramFeatures.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageRep.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/LaRank.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/MultiFeatures.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/RawFeatures.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Sampler.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/Tracker.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/GraphUtils/GraphUtils.cpp"
)

set(Struck_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS}
		CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

add_library(Struck_lib STATIC ${Struck_src})
target_link_libraries(Struck_lib ${OpenCV_LIBS})

set_target_properties(Struck_lib PROPERTIES
		CXX_STANDARD 11
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
)

