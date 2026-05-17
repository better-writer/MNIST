CC = gcc
CFLAGS = -O2 -Wall -lm
TARGET = inference

all: $(TARGET)

$(TARGET): inference.c
	$(CC) $(CFLAGS) -o $(TARGET) inference.c -lm

test: $(TARGET)
	./$(TARGET) mnist_cnn.bin

clean:
	rm -f $(TARGET)

.PHONY: all test clean


