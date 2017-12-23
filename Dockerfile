FROM tensorflow/tensorflow:1.4.1-gpu
RUN pip install \
    numpy
VOLUME /workspace
WORKDIR /workspace
CMD python main.py --phase test --checkpoint_dir ./checkpoint_demo --test_set Set12
