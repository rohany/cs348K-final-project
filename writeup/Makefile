default: main.pdf

main.pdf: main.tex biblio.bib
	pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

clean:
	rm main.aux main.log main.pdf
