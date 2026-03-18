cd /mnt/d/Integration-Game/gesture-trainer-web/python
SIGN_MODEL_PATH=/mnt/d/Integration-Game/AI-models/best_model.pt /home/kiril/miniforge3/bin/conda run -n signs-local python -c "from local_inference_server import app; print(app.title)"
