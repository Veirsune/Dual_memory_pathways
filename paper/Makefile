.PHONY: all chinese architecture training-curves assets audit clean
all:
	latexmk -jobname=GDSM_Paper -pdf -interaction=nonstopmode -halt-on-error main.tex
architecture:
	python scripts/draw_architecture.py
training-curves:
	python scripts/draw_training_curves.py
assets:
	python scripts/build_assets.py
audit: all
	python scripts/audit_manuscript.py
clean:
	latexmk -jobname=GDSM_Paper -c main.tex

chinese:
	latexmk -xelatex -jobname=GDSM_Paper_ZH -interaction=nonstopmode -halt-on-error main_zh.tex
