FROM neubiaswg5/ml-keras-base:latest

RUN git clone https://github.com/matterport/Mask_RCNN.git

RUN cd Mask_RCNN && \
    pip install -r requirements.txt && \
    python setup.py install

ADD https://github.com/Neubias-WG5/W_NucleiSegmentation-MaskRCNN/releases/download/v1.2/weights.h5 /app/weights.h5

RUN chmod 444 /app/weights.h5

ADD wrapper.py /app/wrapper.py

ADD maskrcnn_utils.py /app/maskrcnn_utils.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
