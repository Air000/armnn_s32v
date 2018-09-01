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

### Build Instruction for aarch64 (enable both Tensorflow & Caffe Parser)
```
cd armnn
mkdir build_arm64
cd build_arm64

cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DARMCOMPUTE_ROOT=/path/to/ComputeLibrary -DARMCOMPUTE_BUILD_DIR=/path/to/ComputeLibrary/build  -DBOOST_ROOT=/path/to/dependencies/boost/linux-arm64 -DTF_GENERATED_SOURCES=/path/to/tensorflow-master/_src4arm -DBUILD_TF_PARSER=1 -DPROTOBUF_ROOT=/path/to/dependencies/protobuf/_install_arm-linux/usr/local -DBUILD_TESTS=ON -DARMCOMPUTENEON=ON -DBUILD_CAFFE_PARSER=1 -DCAFFE_GENERATED_SOURCES=/path/to/caffe-master/build_test/include/caffe/proto/caffe.pb.cc

make
```

### Build Instruction for x86 (enable both Tensorflow & Caffe Parser)
```
cd armnn
mkdir build_x86
cd build_x86

cmake .. -DTF_GENERATED_SOURCES=/path/to/dependencies/boost/linux-arm64 -DTF_GENERATED_SOURCES=/path/to/tensorflow-master/_src4arm -DBUILD_TF_PARSER=1 -DPROTOBUF_ROOT=/path/to/dependencies/protobuf/_libx86/usr/local -DBUILD_CAFFE_PARSER=1 -DCAFFE_GENERATED_SOURCES=/path/to/caffe-master/caffe-master/build_test/include -DBUILD_TESTS=ON
```

### Errors & notes:
* armnn/tests/InferenceTestImage.cpp:15:23: fatal error: stb_image.h: No such file or directory
   - If set the flag -DBUILD_TEST=ON, [stb](https://github.com/nothings/stb) is need;  
   - Edit build/tests/CMakeFiles/inferenceTest.dir/flags.make, add -I/path/to/stb to CXX_INCLUDES
* stb_image_write.h:1416:63: error: use of old-style cast [-Werror=old-style-cast]  
   - Edit build/tests/CMakeFiles/inferenceTest.dir/flags.make, delete -Werror in CXX_FLAGS
* protobuf should be compile on x86 first, and then compile again by cross-compiler
* caffe an tensorflow compiled by gcc or cross-compile both are ok, since we only use the xxx.pb.cc and xxx.pb.h in CAFFE/TF_GENERATED_SOURCES
