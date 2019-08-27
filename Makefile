CC=gcc
CFLAGS=-I. 
DEPS = ardbann.h
OBJ = ardbann.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

default: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
