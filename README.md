## Introduction

VerticalMPL is a two-party MPC framework for training machine learning models over vertically partitioned data, based on additive secret sharing. Currently, it is able to support several machine learning models, including linear regression, logistic regression and BP neural network.

## Repository Structure

- `Models/`: Machine learning algorithms: neural networks, linear regression and logistic regression.
- `SCParty/`: The operations required for interactive computing.
- `data/mnist/`: Training dataset  and test dataset.
- `utils/`: Data IO , network IO and matrix operator package. The network is implemented using socket, compatible on both Windows and Ubuntu.
- `CMakeLists.txt`: Define the compile rule for the project. 

## Running

- Install datasets:

    \- `cd VerticalMPL/datasets/mnist`

    \- `python3 download.py`

- Split dataset vertically:

  \- `python3 split.py`

- Install Eigen library:

  -`sudo apt-get install libeigen3-dev`
  
  or download from http://eigen.tuxfamily.org/.
  
  Note that you may have to specify the path to your Eigen library in `CMakeLists.txt`
  
- Specify the platform:

  - In Ubuntu 20.04  (in `utils/Constant.h`):

    ```c++
      `#ifndef UNIX_PLATFORM`
    
      `#define UNIX_PLATFORM`
      
      `#endif`
    ```

  - In Windows 10  (in `CMakeLists.txt`):

    ```cmake
      Add `target_link_libraries(main ws2_32)` to the file.
    ```

- Choose the machine learning model in ` main.cpp `: Uncomment the line of code corresponding to the model:

  ```c++
      /****linear regression****/
      //LR->linear_train(); 
  
      /****logistic regression****/        
      // LR->logistic_train();   
  
      /****neural network****/ 
      // NN->train_model();   
  ```

- Compile the executable file:

  - `cd VerticalMPL`
  - `mkdir build`
  - `cd build `
  - `cmake ..`
  - `make`

- Start two processes and input the party index, respectively:

  - `./main`
  - `Please enter party index:`
  - Enter 0, 1 in order.

## Help

Any question, please contact [20212010090@fudan.edu.cn](mailto:20212010090@fudan.edu.cn).

## Contributor

**Faculty**: Prof. Weili Han

**Students**: Jiaxuan Wang (Graduate Student), Lushan Song (Ph.D Candidate), Xinyu Tu (Graduate Student), 
