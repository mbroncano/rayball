cmake_minimum_required (VERSION 2.6)

project (rayball)

# requires OpenGL and GLUT for visualization
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# add the binary tree to the search path for include files
# so that we will find RayballConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

# add the executable
add_executable(rayball main.cpp)

# libraries for the linker
target_link_libraries(rayball ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ) 

