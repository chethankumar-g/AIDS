from tensorflow.keras.models import load_model
from preprocess import preprocess_image
resnet_model = load_model("models/resnet_model.keras")
vggnet_model = load_model("models/vggnet_model.keras")
xception_model = load_model("models/Xception_model.keras")

