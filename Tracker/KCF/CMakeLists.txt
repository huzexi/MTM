cmake_minimum_required(VERSION 2.8)
project(KCF)

find_package(OpenCV REQUIRED)

set(kcf_src
	"${CMAKE_CURRENT_SOURCE_DIR}/KcfWrapper.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/kcftracker.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/fhog.cpp"
)

add_library(kcf_lib STATIC ${kcf_src})
target_link_libraries(kcf_lib ${OpenCV_LIBS})
set_target_properties(kcf_lib PROPERTIES
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED YES
	CXX_EXTENSIONS NO
)