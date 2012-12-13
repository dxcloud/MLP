DEBUG=no
CC=clang++

ifeq ($(DEBUG),yes)
	CFLAGS=-W -Wall -ansi -pedantic -g
	LDFLAGS=
else
	CFLAGS=-W -Wall -ansi -pedantic
	LDFLAGS=
endif

EXEC=test
SRC= $(wildcard *.cpp)
OBJ= $(SRC:.cpp=.o)

all: $(EXEC)
ifeq ($(DEBUG),yes)
	@echo "Mode debug"
else
	@echo "Mode release"
endif

test: $(OBJ)
	@$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	@$(CC) -o $@ -c $< $(CFLAGS)

.PHONY: clean mrproper

clean:
	@rm -rf *.o

mrproper: clean
	@rm -rf $(EXEC)
