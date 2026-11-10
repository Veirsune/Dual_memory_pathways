.PHONY: all assets audit clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
assets:
	python scripts/build_assets.py
audit: all
	python scripts/audit_manuscript.py
clean:
	latexmk -c
