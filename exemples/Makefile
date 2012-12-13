CLANG = $(shell if command -v clang >/dev/null 2>&1; then echo ok; fi)
ifeq ($(CLANG),ok)
	CC = clang
else
	CC = g++
endif

CFLAGS = -Wall -Werror -O
EXEC_NAME = bank
VPATH = src:src/screens:src/drawable:src/drawable/movable:src/collections
INCLUDES = -isystem ext/SFML/include
OBJ_FILES := main.o menu.o game.o gameE.o
OBJ_FILES += printable.o autoMovable.o gardener.o walker.o banker.o dollar.o
OBJ_FILES += bank.o walkers.o front.o garden.o dollars.o bodies.o

LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
   LIBS = -Lext/SFML/lib64/ -lsfml-graphics -lsfml-window -lsfml-system "-Wl,-rpath,ext/SFML/lib64"
else
   LIBS = -Lext/SFML/lib32/ -lsfml-graphics -lsfml-window -lsfml-system "-Wl,-rpath,ext/SFML/lib32"
endif

all : $(EXEC_NAME)

clean :
	rm $(EXEC_NAME) $(OBJ_FILES) $(OBJ_FILES:.o=.P)

$(EXEC_NAME) : $(OBJ_FILES)
	$(CC) -o $(EXEC_NAME) $(OBJ_FILES) $(LIBS)
	 
$(OBJ_FILES):%.o : %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -o $@ -c $<
	cp $*.d $*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
	rm -f $*.d

-include $(OBJ_FILES:.o=.P)
