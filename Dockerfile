FROM tensorflow/tensorflow:1.0.1-gpu
RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
# 设置root ssh远程登录密码为123456
RUN echo "root:123456" | chpasswd
# 容器需要开放SSH 22端口
EXPOSE 22
RUN pip install \
    numpy

VOLUME /workspace
WORKDIR /workspace
