FROM python:3.9.17-slim-bullseye

WORKDIR /opt/src

COPY requirements.txt .

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

# ENTRYPOINT ["python3", "main.py"]

ENTRYPOINT ["./entrypoint.sh"]
