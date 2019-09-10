CC=gcc
CFLAGS=-I. -lm -O2 -march=native
DEPS = embann.h
OBJ = embann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

default: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
