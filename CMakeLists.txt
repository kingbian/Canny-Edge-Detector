
cmake_minimum_required(VERSION 3.5) #check cmkae version

project(edge_detector.exe) #the name of the project

set(CMAKE_CXX_STANDARD 11) # use the c++ standard

find_package(PkgConfig REQUIRED)

pkg_search_module(OpenCV REQUIRED opencv4)

#include directories 
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_CFLAGS_OTHER})

# Add debug flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_BUILD_TYPE Debug)

set(SOURCES ./src/edge_detector.cpp ./src/main.cpp)

set(HEADERS ./src/edge_detector.h)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} )

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
