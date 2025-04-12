cmake_minimum_required(VERSION 3.17)
project(TFLiteCheck)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/include/)
add_library(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/mac/libtensorflowlite.dylib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES segmentation.cpp)

add_executable(TFLiteCheck ${SOURCES})

target_link_libraries(TFLiteCheck tensorflowlite ${OpenCV_LIBS})
