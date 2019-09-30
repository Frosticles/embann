ifeq ($(OS),Windows_NT)
	CC=gcc
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		CC=gcc
	endif
	ifeq ($(UNAME_S),Darwin)
		CC=gcc-9
	endif
endif

# log regex filter "Analyzing loop at(.|.\n)*LOOP VECTORIZED\n"

LIBS = -lm
CFLAGS = -I. -O3 -march=native -Wall -fopenmp -fverbose-asm -ffast-math -fopt-info-all-optall=opt.log --save-temps #-masm=intel -fopt-info-vec-missed
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
	./cppcheck/cppcheck --addon=cert --addon=./cppcheck/addons/misra.json ./ -i./cppcheck -UARDUINO

debug: CFLAGS+=-g
debug: embann