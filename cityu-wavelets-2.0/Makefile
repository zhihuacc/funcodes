OBJS=main.o unit_test.o src/hammers_and_nails.o src/math_helpers.o src/mat_toolbox.o src/wavelets_toolbox.o app/denoising.o app/commands.o app/inpaint.o

ifeq ($(LIB_HOME),)
LIB_HOME=/usr/local/
endif

ifeq ($(CC),)
CC=g++
endif

ifeq ($(LINKER),)
LINKER=g++
endif


CPP_FLAGS=-g -Wall -O3 -std=c++0x -I$(LIB_HOME)/include
LIBS=-lfftw3 -lopencv_core -lopencv_imgproc -lopencv_highgui -lconfig4cpp
LD_FLAGS=-L$(LIB_HOME)/lib $(LIBS)


ALL:cwt2
	
cwt2:$(OBJS)
	$(LINKER) -o $@  $(OBJS) $(LD_FLAGS)


%.o:%.cpp
	$(CC) -o $@ $(CPP_FLAGS) -c $<

#Be careful to modify this rule, since it may remove files which should NOT be removed.
clean:
	-rm -r $(OBJS) cwt2
	
