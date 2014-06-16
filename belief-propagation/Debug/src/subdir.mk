################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/dynamical_nested_loops.cpp \
../src/factor_graph.cpp \
../src/graph_struct.cpp \
../src/grid_mrf.cpp 

OBJS += \
./src/dynamical_nested_loops.o \
./src/factor_graph.o \
./src/graph_struct.o \
./src/grid_mrf.o 

CPP_DEPS += \
./src/dynamical_nested_loops.d \
./src/factor_graph.d \
./src/graph_struct.d \
./src/grid_mrf.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


