FROM python:3.11

COPY . .

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install .

WORKDIR /app/work/
COPY mlex_pytorch_autoencoders/ mlex_pytorch_autoencoders/
CMD ["bash"]
