cmake_minimum_required(VERSION 2.8)
project(DSST)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# add OpenCV
set(OPENCV_DIR_HINT "/home/zeke/Libs/opencv/opencv-3.4.3/build")
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})

# include
set(DSST_include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(DSST_include
        "${DSST_include_path}/dsst_tracker.hpp"
        "${DSST_include_path}/dsst_debug.hpp"
        "${DSST_include_path}/TrackTask.h"
        "${DSST_include_path}/DsstWrapper.h"
)

# src
set(DSST_src_path "${CMAKE_CURRENT_SOURCE_DIR}/src/")
set(DSST_src
        "${DSST_src_path}/TrackTask.cpp"
        "${DSST_src_path}/DsstWrapper.cpp"
        "${DSST_src_path}/math_helper.cpp"
)

# add 3rdparty source
set(CF_TCLAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tclap/")
set(CF_CV_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cv_ext/")
set(CF_PIOTR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/piotr/")
set(CF_PIOTR_SOURCES
        ${CF_PIOTR_DIR}/gradientMex.cpp)
set(CF_LIB_COMMON_SOURCES
        ${CF_CV_EXT_DIR}/shift.cpp
        ${CF_CV_EXT_DIR}/math_spectrums.cpp
        ${CF_PIOTR_SOURCES})

add_library(DSST_lib ${DSST_src} ${CF_PIOTR_SOURCES} ${CF_LIB_COMMON_SOURCES})
target_include_directories(DSST_lib PRIVATE ${DSST_include_path} ${CF_CV_EXT_DIR} ${CF_PIOTR_DIR})
target_link_libraries(DSST_lib ${OpenCV_LIBS})

set_target_properties(DSST_lib PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)