cmake_minimum_required(VERSION 2.8)
project(CT)

find_package(OpenCV REQUIRED)

set(ct_src
	"${CMAKE_CURRENT_SOURCE_DIR}/CtWrapper.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/CompressiveTracker.cpp"
)

add_library(ct_lib STATIC ${ct_src})
target_link_libraries(ct_lib ${OpenCV_LIBS})
set_target_properties(ct_lib PROPERTIES
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED YES
	CXX_EXTENSIONS NO
)