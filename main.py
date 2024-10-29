import json
import base64
import io
from PIL import Image
import yaml
from ultralytics import YOLO
import supervision as sv

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # Load YOLOv8 model
    context.user_data.model = YOLO("wood.pt")
    context.user_data.labels = labels

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run YOLOv8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    
    image = Image.open(buf)
    results = context.user_data.model(image, conf=threshold)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    response = []
    if len(detections) > 0:
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            
            response.append({
                "confidence": str(confidence),
                "label": context.user_data.labels.get(class_id, "wood"),
                "points": xyxy.tolist(),
                "type": "rectangle"
            })

    return context.Response(body=json.dumps(response), 
                          headers={},
                          content_type='application/json',
                          status_code=200)