# Start FROM Nvidia Tensorrt image
FROM nvcr.io/nvidia/tensorrt:21.09-py3

# Install linux packages
RUN apt update && apt install -y libgl1-mesa-dev

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache -r requirements.txt && pip install --no-cache jupyter
RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch-quantization toolkit
RUN pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

# Create working directory
RUN mkdir -p /root/space/projects
WORKDIR /root/space/projects