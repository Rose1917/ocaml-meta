epoch=1
.PHONY: test gen
main:
	dune exec ./main.exe ${epoch}
clean:
	dune clean
test:
	dune runtest
count:
	find -path ./_build -prune  -o \( -name '*.ml' -o -name '*.c' -o -name '*.h' \) -print|tr '\n' ' '|xargs wc -l
