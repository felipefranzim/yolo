# train_debug_run.py
import os, time, traceback
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONUNBUFFERED"] = "1"

from ultralytics import YOLO
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("ultralytics").setLevel(logging.DEBUG)

if __name__ == "__main__":
    try:
        print("=== DEBUG TRAIN START ===")
        print("start time:", time.strftime("%Y-%m-%d %H:%M:%S"))
        model = YOLO('yolov8n.pt')
        print("model loaded OK")
        # Parâmetros explícitos; ajuste epochs para 3 para testar
        res = model.train(
            data='coco128.yaml',
            epochs=3,
            imgsz=640,
            batch=1,
            device=0,
            workers=0,       # evitar multiprocessing issues no Windows
            plots=False,
            show=False,
            save_period=1,
            verbose=True,
            amp=False        # desativa AMP checks adicionais para teste
        )
        print("train() retornou:", type(res), res)
        print("end time:", time.strftime("%Y-%m-%d %H:%M:%S"))
        print("=== DEBUG TRAIN END ===")
    except Exception:
        print("EXCEPTION during train():")
        traceback.print_exc()