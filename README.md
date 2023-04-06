# TabularDataExtraction

Initial attempt at creating a tabular data extraction tool from scanned PDFs. 

Dependencies

conda install paddlepaddle==2.4.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ GPU doesn’t work for me
Brew install poppler
Pip install pdf2image - seems working
Pip install tensorflow
Pip install opencv-python
pip install "paddleocr>=2.6.0.3"
Pip install protobuf==3.20.0
wget wget https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.wh
pip install -U layoutparser-0.0.0-py3-none-any.whl

Plus standard anaconda installation.
