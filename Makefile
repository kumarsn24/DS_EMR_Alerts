.PHONY: install run test
install:
	pip install -r requirements.txt
run:
	uvicorn src.backend_emralerts.api.app:app --reload --port 8000
test:
	pytest -q
