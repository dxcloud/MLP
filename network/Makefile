CC = g++
CFLAGS = -Wall -Werror -O
EXEC_NAME = networking
OBJ_FILES := main.o network.o

all : $(EXEC_NAME)

clean :
	rm $(EXEC_NAME) $(OBJ_FILES)
	
$(EXEC_NAME) : $(OBJ_FILES)
	$(CC) -o $(EXEC_NAME) $(OBJ_FILES)
	 
$(OBJ_FILES):%.o : %.cpp
	$(CC) $(CFLAGS) -MMD -o $@ -c $<
	cp $*.d $*.P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $*.P; \
	rm -f $*.d

-include $(OBJ_FILES:.o=.P)
