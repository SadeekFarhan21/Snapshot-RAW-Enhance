.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib opt2026.cls jmlr.cls jmlrutils.sty ../../figures/identifiability_gate.png
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -pdf -C main.tex
