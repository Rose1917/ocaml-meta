main:
	dune exec ./main.exe
test:
	time dune exec ./main.exe
clean:
	dune clean
count:
	find -path ./_build -prune  -o -name '*.ml' -print|tr '\n' ' '|xargs wc -l
