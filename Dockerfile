FROM tensorflow/tensorflow:1.0.1-gpu
RUN pip install \
    numpy
VOLUME /workspace
WORKDIR /workspace
CMD    ["/usr/sbin/sshd", "-D"]
