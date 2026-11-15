.PHONY: all architecture assets audit clean
all:
	latexmk -jobname=GDSM_Paper -pdf -interaction=nonstopmode -halt-on-error main.tex
architecture:
	python scripts/draw_architecture.py
assets:
	python scripts/build_assets.py
audit: all
	python scripts/audit_manuscript.py
clean:
	latexmk -jobname=GDSM_Paper -c main.tex
