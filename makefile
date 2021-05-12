.PHONY: test
main:
	dune exec ./main.exe
clean:
	dune clean
test:
	dune runtest
count:
	find -path ./_build -prune  -o \( -name '*.ml' -o -name '*.c' -o -name '*.h' \) -print|tr '\n' ' '|xargs wc -l
