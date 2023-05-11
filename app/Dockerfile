FROM python:3.9

# Crear un directorio de trabajo
RUN mkdir /app

# Crear un directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos al contenedor
COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENV AWS_ACCESS_KEY_ID <AWS_ACCESS_KEY_ID>
ENV AWS_SECRET_ACCESS_KEY <AWS_ACCESS_KEY>

EXPOSE 8506
CMD ["streamlit", "run", "--server.port", "8506","--server.address", "0.0.0.0", "app.py"]
