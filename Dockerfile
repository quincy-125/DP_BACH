FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest
# Enable installing software when building from a DLVM:
COPY CertEmulationCA.crt /usr/local/share/ca-certificates
COPY sources.list /etc/apt/sources.list
# The steps below with libgdk-pixbuf are to avoid a dependency error that was
# preventing the installation of openslide-tools, which are required for 
# OpenSlide, as deocumented at https://openslide.org/download/
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y openslide-tools

RUN pip install --no-cache-dir \
  google-cloud-storage==1.44.0 \
  openslide-python==1.1.2 \
  scikit_image==0.18.3 \
  seaborn==0.11.2 

WORKDIR /opt/app
# COPY entrypoint.sh ./
COPY * ./

# ENTRYPOINT ["./entrypoint.sh"]
ENTRYPOINT ["python", "main.py"]