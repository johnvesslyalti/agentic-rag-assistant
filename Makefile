.PHONY: install test run ui ingest smoke check-env docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest --tb=short -q

run:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

ingest:
	python scripts/ingest.py

smoke:
	python scripts/smoke_test.py

check-env:
	python scripts/check_env.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
