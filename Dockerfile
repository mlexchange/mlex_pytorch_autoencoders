FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        tree
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app/work/
COPY src/ src/
CMD ["bash"]