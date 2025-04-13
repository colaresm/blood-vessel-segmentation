cmake_minimum_required(VERSION 3.17)
project(SegmentBloodVessels)
set(CMAKE_CXX_STANDARD 14)

# OpenCV
if(WIN32)
    set(OpenCV_DIR C:/opencv/opencv4/build)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/include/)

set(SOURCES main.cpp)
add_executable(SegmentBloodVessels ${SOURCES})

# TensorFlow Lite configuration
if(APPLE)
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/mac/libtensorflowlite.dylib)
    target_link_libraries(SegmentBloodVessels PRIVATE tensorflowlite ${OpenCV_LIBS})
endif()

if(WIN32)
    find_library(TFLITE tensorflowlite.dll HINTS ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/windows_x64/)
    find_library(FLATBUFFERS flatbuffers HINTS ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/windows_x64/)
    target_link_libraries(SegmentBloodVessels PRIVATE ${TFLITE} ${FLATBUFFERS} ${OpenCV_LIBS})

    add_custom_command(
        TARGET SegmentBloodVessels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
    )

    add_custom_command(
        TARGET SegmentBloodVessels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenCV_DIR}/x64/vc15/bin/opencv_world452.dll ${CMAKE_BINARY_DIR}/Release/
    )
endif()
