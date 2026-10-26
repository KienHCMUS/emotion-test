cmake_minimum_required(VERSION 3.10)
project(MyOpenCVProject)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(MyApp main.cpp)

# Link OpenCV libraries
target_link_libraries(MyApp ${OpenCV_LIBS})

# Copy .onnx files to build directory for runtime access
add_custom_command(TARGET MyApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/models/backbone.onnx"
    "$<TARGET_FILE_DIR:MyApp>/backbone.onnx"
)
add_custom_command(TARGET MyApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/models/neckhead.onnx"
    "$<TARGET_FILE_DIR:MyApp>/neckhead.onnx"
)