.PHONY: test
main:
	dune exec ./main.exe
clean:
	dune clean
test:
	dune runtest
count:
	find -path ./_build -prune  -o -name '*.ml' -print|tr '\n' ' '|xargs wc -l
