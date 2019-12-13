cmake_minimum_required (VERSION 3.4.1)

project(bert-cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
file(GLOB DIR_SRCS src/*.cc)

include(FindProtobuf)
find_package(Protobuf REQUIRED)

include(FindMKL)
find_package(MKL REQUIRED)

find_package(Threads REQUIRED)

set(utf8proc_LIBRARY /usr/local/lib/libutf8proc.a)
set(utf8proc_INCLUDE_DIR /usr/local/include)

include_directories(${utf8proc_INCLUDE_DIR})

add_library(bertcpp SHARED ${DIR_SRCS})
target_link_libraries(bertcpp ${PROTOBUF_LIBRARY} ${MKL_LIBRARIES} ${utf8proc_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

file(GLOB TEST_SRCS test/*.cpp)
add_executable(bert_test ${TEST_SRCS})
target_link_libraries(bert_test bertcpp ${GTEST_BOTH_LIBRARIES})
add_test(unitest bert_test)
