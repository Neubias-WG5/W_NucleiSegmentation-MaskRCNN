FROM neubiaswg5/ml-keras-base:latest

RUN git clone https://github.com/matterport/Mask_RCNN.git

RUN cd Mask_RCNN && \
    mkdir logs && \
    pip install -r requirements.txt && \
    python setup.py install

ADD weights.h5 /Mask_RCNN/logs/weights.h5

ADD wrapper.py /app/wrapper.py

ADD maskrcnn_utils.py /app/maskrcnn_utils.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
