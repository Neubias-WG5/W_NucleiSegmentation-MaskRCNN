FROM python:3.6.9-stretch

# ------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.3.0.poc.1 && pip install . && \
    rm -r /Cytomine-python-client

# ------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.8 && pip install .

# install utilities binaries
RUN chmod +x /neubiaswg5-utilities/bin/*
RUN cp /neubiaswg5-utilities/bin/* /usr/bin/ && \
    rm -r /neubiaswg5-utilities

# ------------------------------------------------------------------------------

RUN pip install scikit-learn keras h5py joblib
RUN pip install tensorflow==1.13.1

RUN git clone https://github.com/matterport/Mask_RCNN.git

RUN cd Mask_RCNN && \
    pip install -r requirements.txt && \
    python setup.py install

RUN pip install gdown

RUN mkdir /app && \
    cd /app && \
    gdown https://drive.google.com/uc?id=19EmZ57LXSArG-Z1HC8NOtrQUxGNzW3vv

RUN chmod 444 /app/weights.h5

ADD wrapper.py /app/wrapper.py

ADD maskrcnn_utils.py /app/maskrcnn_utils.py

ENTRYPOINT ["python3.6","/app/wrapper.py"]
