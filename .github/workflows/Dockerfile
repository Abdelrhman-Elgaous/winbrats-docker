FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /mlcube_project/requirements.txt
RUN pip3 install --no-cache-dir \
      torch==2.1.2 torchvision==0.16.2 \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && pip3 install --no-cache-dir -r /mlcube_project/requirements.txt \
 && python3 -c "import torch, numpy; from scipy import ndimage; print('torch', torch.__version__, '| numpy', numpy.__version__, '| scipy.ndimage OK')"
COPY . /mlcube_project
COPY weights /opt/weights
ENV nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results \
    nnUNet_compile=F \
    MKL_THREADING_LAYER=GNU
ENTRYPOINT ["python3", "/mlcube_project/run_bratsmets.py"]
