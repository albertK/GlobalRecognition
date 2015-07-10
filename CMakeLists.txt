cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(recognition)

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(trainDB src/trainDB.cpp)
target_link_libraries(trainDB ${PCL_LIBRARIES})

#TEST
add_executable(pre_processing_test src/test/pre_processing_test.cpp)
target_link_libraries(pre_processing_test ${PCL_LIBRARIES})

add_executable(descriptor_estimation_test src/test/descriptor_estimation_test.cpp)
target_link_libraries(descriptor_estimation_test ${PCL_LIBRARIES})

add_executable(recognition_database_test src/test/recognition_database_test.cpp)
target_link_libraries(recognition_database_test ${PCL_LIBRARIES})

add_executable(pose_estimation_test src/test/pose_estimation_test.cpp)
target_link_libraries(pose_estimation_test ${PCL_LIBRARIES})

add_executable(post_processing_test src/test/post_processing_test.cpp)
target_link_libraries(post_processing_test ${PCL_LIBRARIES})

add_executable(global_recognition_test src/test/global_recognition_test.cpp)
target_link_libraries(global_recognition_test ${PCL_LIBRARIES})