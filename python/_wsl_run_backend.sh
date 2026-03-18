cd /mnt/d/Integration-Game/gesture-trainer-web/python
export SIGN_MODEL_PATH=/mnt/d/Integration-Game/AI-models/best_model.pt
nohup /home/kiril/miniforge3/envs/signs-local/bin/python -m uvicorn local_inference_server:app --host 127.0.0.1 --port 8000 >/tmp/signs-local-backend.log 2>&1 &
echo $!
