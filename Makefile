CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -Wall -Wextra -Wpedantic
LDLIBS  = -lm

SRC     = $(wildcard src/*.c)
OBJ     = $(SRC:src/%.c=build/%.o)
BIN     = mnist-nn

.PHONY: all run data clean debug

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

build/%.o: src/%.c src/*.h | build
	$(CC) $(CFLAGS) -c -o $@ $<

build:
	mkdir -p build

# Download MNIST and convert it to the CSV format the program expects.
data:
	python3 scripts/download_mnist.py

# Train the default model (784 -> 64 -> 64 -> 10, 30 epochs, lr 0.1).
run: $(BIN)
	./$(BIN)

# Build with AddressSanitizer / UBSan for checking memory safety.
debug: CFLAGS = -std=c11 -O1 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: clean $(BIN)

clean:
	rm -rf build $(BIN)
