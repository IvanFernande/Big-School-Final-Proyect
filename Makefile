env:
	python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt
prep:
	python -m src.data_prep
train:
	python -m src.train
eval:
	python -m src.evaluate
biz:
	python -m src.business_metrics
serve:
	uvicorn src.serve_api:app --reload
