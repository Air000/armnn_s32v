# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test

# Include any dependencies generated for this target.
include CMakeFiles/armnnUtils.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/armnnUtils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/armnnUtils.dir/flags.make

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o: CMakeFiles/armnnUtils.dir/flags.make
CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o: ../src/armnnUtils/Logging.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o -c /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Logging.cpp

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.i"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Logging.cpp > CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.i

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.s"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Logging.cpp -o CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.s

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.requires:

.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.requires

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.provides: CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.requires
	$(MAKE) -f CMakeFiles/armnnUtils.dir/build.make CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.provides.build
.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.provides

CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.provides.build: CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o


CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o: CMakeFiles/armnnUtils.dir/flags.make
CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o: ../src/armnnUtils/Permute.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o -c /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Permute.cpp

CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.i"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Permute.cpp > CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.i

CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.s"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/Permute.cpp -o CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.s

CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.requires:

.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.requires

CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.provides: CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.requires
	$(MAKE) -f CMakeFiles/armnnUtils.dir/build.make CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.provides.build
.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.provides

CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.provides.build: CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o


CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o: CMakeFiles/armnnUtils.dir/flags.make
CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o: ../src/armnnUtils/DotSerializer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o -c /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/DotSerializer.cpp

CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.i"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/DotSerializer.cpp > CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.i

CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.s"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/DotSerializer.cpp -o CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.s

CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.requires:

.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.requires

CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.provides: CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.requires
	$(MAKE) -f CMakeFiles/armnnUtils.dir/build.make CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.provides.build
.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.provides

CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.provides.build: CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o


CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o: CMakeFiles/armnnUtils.dir/flags.make
CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o: ../src/armnnUtils/HeapProfiling.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o -c /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/HeapProfiling.cpp

CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.i"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/HeapProfiling.cpp > CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.i

CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.s"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/HeapProfiling.cpp -o CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.s

CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.requires:

.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.requires

CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.provides: CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.requires
	$(MAKE) -f CMakeFiles/armnnUtils.dir/build.make CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.provides.build
.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.provides

CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.provides.build: CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o


CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o: CMakeFiles/armnnUtils.dir/flags.make
CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o: ../src/armnnUtils/LeakChecking.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o -c /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/LeakChecking.cpp

CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.i"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/LeakChecking.cpp > CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.i

CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.s"
	/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/worktools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/src/armnnUtils/LeakChecking.cpp -o CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.s

CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.requires:

.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.requires

CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.provides: CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.requires
	$(MAKE) -f CMakeFiles/armnnUtils.dir/build.make CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.provides.build
.PHONY : CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.provides

CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.provides.build: CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o


# Object files for target armnnUtils
armnnUtils_OBJECTS = \
"CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o" \
"CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o" \
"CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o" \
"CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o" \
"CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o"

# External object files for target armnnUtils
armnnUtils_EXTERNAL_OBJECTS =

libarmnnUtils.a: CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/build.make
libarmnnUtils.a: CMakeFiles/armnnUtils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libarmnnUtils.a"
	$(CMAKE_COMMAND) -P CMakeFiles/armnnUtils.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/armnnUtils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/armnnUtils.dir/build: libarmnnUtils.a

.PHONY : CMakeFiles/armnnUtils.dir/build

CMakeFiles/armnnUtils.dir/requires: CMakeFiles/armnnUtils.dir/src/armnnUtils/Logging.cpp.o.requires
CMakeFiles/armnnUtils.dir/requires: CMakeFiles/armnnUtils.dir/src/armnnUtils/Permute.cpp.o.requires
CMakeFiles/armnnUtils.dir/requires: CMakeFiles/armnnUtils.dir/src/armnnUtils/DotSerializer.cpp.o.requires
CMakeFiles/armnnUtils.dir/requires: CMakeFiles/armnnUtils.dir/src/armnnUtils/HeapProfiling.cpp.o.requires
CMakeFiles/armnnUtils.dir/requires: CMakeFiles/armnnUtils.dir/src/armnnUtils/LeakChecking.cpp.o.requires

.PHONY : CMakeFiles/armnnUtils.dir/requires

CMakeFiles/armnnUtils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/armnnUtils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/armnnUtils.dir/clean

CMakeFiles/armnnUtils.dir/depend:
	cd /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test /media/air/6f43c45d-eb70-eb42-801a-ed77dc73e87f/armnn/armnn/build_arm64_test/CMakeFiles/armnnUtils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/armnnUtils.dir/depend

