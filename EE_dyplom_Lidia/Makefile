pdflatex : clean
	pdflatex -interaction=nonstopmode EE-dyplom
	biber EE-dyplom
	makeglossaries EE-dyplom
	pdflatex -interaction=nonstopmode EE-dyplom
	pdflatex -interaction=nonstopmode EE-dyplom

	
xelatex : clean
	xelatex -interaction=nonstopmode EE-dyplom
	biber EE-dyplom
	makeglossaries EE-dyplom
	xelatex -interaction=nonstopmode EE-dyplom
	xelatex -interaction=nonstopmode EE-dyplom

clean :
	-rm *.acn *.app *.aux *.bbl *.blg *.bcf *.glo *.ist *.lof *.log *.lot *.out *.pdf *.run.xml *.slo *.toc
