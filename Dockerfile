 FROM ubuntu:18.04
 RUN pip3 install -r requirements.txt
 CMD python3 ./app.py
