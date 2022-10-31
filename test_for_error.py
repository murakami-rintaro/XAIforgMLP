import timm
from PIL import Image
from torchvision import transforms
import timm.models.mlp_mixer


#モデル作成
model = timm.create_model("gmlp_s16_224", pretrained=True)
model.eval()
print("model")

transform = transforms.Compose(
    [
        transforms.Resize(256),  # (256, 256) で切り抜く。
        transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
        transforms.ToTensor(),  # テンソルにする。
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する。
    ]
)

img = Image.open("cat.jpg")
print(type(img))
inputs = transform(img)
inputs = inputs.unsqueeze(0)
print(type(inputs))
import PixelAblationCAMfrogMLP
exp = PixelAblationCAMfrogMLP.PixelAblationCAMfrogMLP(model)
exp_out = exp.calc_value_las_block_beta(inputs)
print(exp_out)