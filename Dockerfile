# Use a python image as the base
FROM python:3.9

RUN mkdir -p /app

RUN mkdir -p ./models

# Set the working directory
WORKDIR /app



# Copy the requirements to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory contents to the working directory
COPY ./app /app

# Copy the models into working directory
COPY ./models ./models


RUN ls -R /app

# Run the FastAPI application using Uvicorn
CMD ["fastapi", "run", "main.py", "--port", "80"]