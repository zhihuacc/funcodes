OBJS=src/hammers_and_nails.o src/math_helpers.o src/mat_toolbox.o src/wavelets_toolbox.o app/denoising.o app/commands.o app/inpaint.o

ifeq ($(LIB_HOME),)
LIB_HOME=/usr/local
endif

ifeq ($(OS),)
CC=g++
LINKER=g++
STD_FLAG=-std=c++0x
endif

ifeq ($(OS),Linux)
CC=g++
LINKER=g++
STD_FLAG=-std=c++0x
endif

ifeq ($(OS),MacOS)
CC=clang++
LINKER=clang++
STD_FLAG=-std=c++11
endif


ifeq ($(DIR),)
DIR=/usr/local
endif


CPP_FLAGS=-g -Wall -O3 $(STD_FLAG) -I$(LIB_HOME)/include
LIBS=-lfftw3 -lopencv_core -lopencv_imgproc -lopencv_highgui -lconfig4cpp
LD_FLAGS=-L$(LIB_HOME)/lib $(LIBS)


ALL:lib/libtpctf.so bin/tpctf2 

lib/libtpctf.so:$(OBJS)
	mkdir -p lib
	$(LINKER) -o $@ $(OBJS) $(LD_FLAGS) -fpic -shared
	
bin/tpctf2:main.o unit_test.o
	mkdir -p bin
	$(LINKER) -o $@  $(OBJS) main.o unit_test.o $(LD_FLAGS) -L./lib -ltpctf

main.o:main.cpp
	$(CC) -o $@ $(CPP_FLAGS) -c $<

unit_test.o:unit_test.cpp
	$(CC) -o $@ $(CPP_FLAGS) -c $<

%.o:%.cpp
	$(CC) -o $@ $(CPP_FLAGS) -fpic -c $<

#Be careful to modify this rule, since it may remove files which should NOT be removed.
clean:
	-rm -r $(OBJS) main.o unit_test.o lib/libtpctf.so bin/tpctf2

install:
	mkdir -p $(DIR)/include/tpctf
	mkdir -p $(DIR)/lib
	mkdir -p $(DIR)/bin
	cp -v include/* $(DIR)/include/tpctf
	cp -v lib/libtpctf.so $(DIR)/lib
	cp -v bin/tpctf2 $(DIR)/bin
	
