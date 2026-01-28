安装依赖
pip install -r requirements.txt

启动服务
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

浏览器打开
http://localhost:8000