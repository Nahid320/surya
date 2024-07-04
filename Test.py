from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings

if __name__ == "__main__":
    IMAGE_PATH = "C:/BitBucketRepo/datascienceprojects/ChartExt/P4/page4.png"
    image = Image.open(IMAGE_PATH)
    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    # layout_predictions is a list of dicts, one per image
    line_predictions = batch_text_detection([image], det_model, det_processor)
    layout_predictions = batch_layout_detection([image], model, processor, line_predictions)