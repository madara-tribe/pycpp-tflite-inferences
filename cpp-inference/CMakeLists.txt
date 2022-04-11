cmake_minimum_required(VERSION 2.8)
project(inference)

# Create Main project
add_executable(inference
    inference.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(inference PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(inference ${OpenCV_LIBS})
endif()

# For Tensorflow Lite
target_link_libraries(inference ${PROJECT_SOURCE_DIR}/tensorflow_utils/tf_prebuild_linux-cpu-x86_64/libtensorflowlite.so)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/eigen )
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/absl)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src)
target_include_directories(inference PUBLIC ${PROJECT_SOURCE_DIR}/tensorflow_utils/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++11 -lstdc++")

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")
