FROM python:3.8

LABEL maintainer="Jonathan Elejalde https://github.com/JonathanElejalde"

ADD requirements.txt /

RUN pip install -r requirements.txt

# TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install \
  && pip install ta-lib

RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz


