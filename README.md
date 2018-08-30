# Arm NN

For more information about Arm NN, see: <https://developer.arm.com/products/processors/machine-learning/arm-nn>

There is a getting started guide here using TensorFlow: <https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-tensorflow>

There is a getting started guide here using Caffe: <https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-caffe>

### Build Instructions

Arm tests the build system of Arm NN with the following build environments:

* Android NDK: [How to use Android NDK to build ArmNN](BuildGuideAndroidNDK.md)
* Cross compilation from x86_64 Ubuntu to arm64 Linux
* Native compilation under arm64 Debian 9

Arm NN is written using portable C++14 and the build system uses [CMake](https://cmake.org/) so it is possible to build for a wide variety of target platforms, from a wide variety of host environments.

### Build Instruction for aarch64
```
cd armnn
mkdir build_arm64
cd build_arm64

cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DARMCOMPUTE_ROOT=/path/to/ComputeLibrary -DARMCOMPUTE_BUILD_DIR=/path/to/ComputeLibrary/build  -DBOOST_ROOT=/path/to/dependencies/boost/linux-arm64 -DTF_GENERATED_SOURCES=/path/to/tensorflow-master/_src4arm -DBUILD_TF_PARSER=1 -DPROTOBUF_ROOT=/path/to/dependencies/protobuf/_install_arm-linux/usr/local -DBUILD_TESTS=ON -DARMCOMPUTENEON=ON

make
```
