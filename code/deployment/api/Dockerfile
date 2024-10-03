FROM python:3.8

WORKDIR /api

COPY ./models /api/models


# Copy the main FastAPI app file into the working directory
COPY ./code/deployment/api/main.py /api/main.py

COPY ./code/deployment/api/requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir -r /api/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

