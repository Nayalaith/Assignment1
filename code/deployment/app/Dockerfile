FROM python:3.8

WORKDIR /app

COPY ./code/deployment/app/main.py /app/main.py

COPY ./code/deployment/app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

