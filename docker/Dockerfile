FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN useradd -ms /bin/bash algorithm


USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python3 -m pip install --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY --chown=algorithm:algorithm model/ /opt/algorithm/model/
COPY --chown=algorithm:algorithm test/ /opt/algorithm/test/
COPY --chown=algorithm:algorithm model_weights/ /opt/algorithm/model_weights/
COPY --chown=algorithm:algorithm segment_anything/ /opt/algorithm/segment_anything/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm sam_lora_image_encoder.py /opt/algorithm/
COPY --chown=algorithm:algorithm segmentation.py /opt/algorithm/


ENTRYPOINT python3 -m process $0 $@

LABEL nl.diagnijmegen.rse.algorithm.name=seg_algorithm


