cmake_minimum_required(VERSION 3.15)
project(VerticalMPL)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -maes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -pthread -fopenmp")

find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIRS}) #specify the path to your Eigen library
set(SOURCE_FILES_BASE  Models/NNModel.cpp Models/NNModel.h Models/LRModel.cpp Models/LRModel.h SCParty/Functionalities.cpp SCParty/Functionalities.h SCParty/PreCompute.cpp SCParty/PreCompute.h utils/SocketManager.cpp utils/SocketManager.h utils/SocketOnline.cpp utils/SocketOnline.h utils/IOManager.cpp utils/IOManager.h  utils/Mat.cpp utils/Mat.h utils/Constant.cpp utils/Constant.h)

add_executable(main main.cpp Models/NNModel.cpp Models/NNModel.h Models/LRModel.cpp Models/LRModel.h SCParty/Functionalities.cpp SCParty/Functionalities.h SCParty/PreCompute.cpp SCParty/PreCompute.h utils/SocketManager.cpp utils/SocketManager.h utils/SocketOnline.cpp utils/SocketOnline.h utils/IOManager.cpp utils/IOManager.h utils/Mat.cpp utils/Mat.h utils/Constant.cpp utils/Constant.h)
# target_link_libraries(main ws2_32)
add_definitions("-Wall -g")