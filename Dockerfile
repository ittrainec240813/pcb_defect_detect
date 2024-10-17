FROM python:3.10

COPY . ./pcb_defect_detect

WORKDIR /

# fix "ImportError: libGL.so.1: cannot open shared object file: No such file or directory"
# a cv2 issue
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /pcb_defect_detect/requirements.txt

CMD ["uvicorn", "pcb_defect_detect.main:app", "--host", "0.0.0.0", "--port", "7860"]