ifeq ($(OS),Windows_NT)
	CC=gcc
	CPP_CHECK=./cppcheck/cppcheck.exe
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CC=gcc
		CPP_CHECK=./cppcheck/cppcheck
	endif
	ifeq ($(UNAME_S),Darwin)
		CC=gcc-9
		CPP_CHECK=./cppcheck/cppcheck-mac
	endif
endif

# log regex filter "Analyzing loop at(.|.\n)*LOOP VECTORIZED\n"

LIBS = -lm
CFLAGS = -I. -O3 -march=native -Wall -fopenmp -flto -fverbose-asm -fopt-info-all-optall=opt.log --save-temps #-masm=intel -fopt-info-vec-missed -ffast-math
DEPS = embann.h
OBJ = embann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

embann: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean check debug

clean:
	rm -f ./*.o ./*.s ./*.i ./*.c.dump ./embann ./opt.log

check:
	CPP_CHECK --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

debug: CFLAGS+=-g
debug: embann