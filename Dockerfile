FROM python:3.8

RUN apt-get update && apt-get upgrade
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /usr/src/ufc/
WORKDIR /usr/src/ufc/
COPY ./ .

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]