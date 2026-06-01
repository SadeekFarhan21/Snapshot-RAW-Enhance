.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -f -pdflatex='pdflatex -interaction nonstopmode' -pdf poster.tex || test -f poster.pdf

clean:
	latexmk -pdf -C
