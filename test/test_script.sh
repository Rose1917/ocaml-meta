#!/bin/sh
res_text="test_res.txt"
touch ${res_text}
dune build ./main.exe
for i in $(seq 10)
do
	time dune exec ./main.exe >>${res_text}
done
