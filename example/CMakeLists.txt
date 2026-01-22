cmake_minimum_required(VERSION 3.0)

PROJECT(mnist_example)
# Usa C++17 o el est�ndar que necesites
set(CMAKE_CXX_STANDARD 17)

# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.

find_package(MNIST PATHS ..)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})

# Buscar paquetes instalados con vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)

add_executable(mnist_example main.cpp)

# Add modular sources
target_sources(mnist_example PRIVATE
    som.cpp
    visualizer.cpp
)

# Ensure example directory is on the include path so headers are found
target_include_directories(mnist_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Linkear librer�as encontradas
target_link_libraries(mnist_example
    PRIVATE
    glfw
    glm::glm
    ${OpenCV_LIBS}
    GLEW::GLEW
)

# Keep compile features
target_compile_features(mnist_example PRIVATE cxx_range_for)
# Pass MNIST data directory to main.cpp
target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")





