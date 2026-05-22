$bibtex_use = 2;
$clean_ext  = "nav snm";

# Thayer poster theme uses fontspec (Lato, Raleway) which requires
# lualatex or xelatex. Force lualatex regardless of how latexmk is
# invoked (Makefile, IDE, command line).
$pdf_mode  = 4;          # 4 = lualatex
$lualatex  = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
