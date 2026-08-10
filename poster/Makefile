.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -pdflatex='pdflatex -interaction nonstopmode -halt-on-error' -pdf poster.tex

clean:
	latexmk -pdf -C
