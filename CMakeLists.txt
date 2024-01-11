
cmake_minimum_required(VERSION 3.5) #check cmkae version

project(edge_detector.exe) #the name of the project

set(CMAKE_CXX_STANDARD 11) # use the c++ standard

find_package(PkgConfig REQUIRED)

pkg_search_module(OpenCV REQUIRED opencv4)

#include directories 
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_CFLAGS_OTHER})


set(SOURCES edge_detector.cpp main.cpp)

set(HEADERS edge_detector.h)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES} )

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})