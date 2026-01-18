FROM python:3.12.3-slim

WORKDIR /app

RUN pip install pipenv 

COPY Pipfile.lock Pipfile .

RUN pipenv install --system --deploy --ignore-pipfile

COPY . .

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host","0.0.0.0", "--port", "9696"]