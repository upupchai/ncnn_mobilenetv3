import torch
#from model import MobileNetV2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from mobilenetv3 import MobileNetV3,mobilenetv3_large
import numpy as np
import torch
import torchvision
import torch.utils.model_zoo
import torch.onnx
# import caffe2.python.onnx.backend as backend
# import netron
from model_v2 import MobileNetV2

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("/media/yiji/ea1def5e-614d-46de-ae59-45f647ac3d2a1/lagopus/PycharmProject/IMAGEclassfier/mobilenetv3.pytorch/5left.png")
img = img.convert('RGB')
plt.imshow(img)

print()
# [N, C, H, W]
img = data_transform(img)
print(type(img))
# expand batch dimensionss
img = torch.unsqueeze(img, dim=0)


print(img.shape)
print(type(img))
# input_names  = ["img"]
# output_names = ["y"]

# x = imgq
# read class_indict


try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
# model = MobileNetV2(num_classes=12).to()
# model = MobileNetV2(num_classes=8)
model = mobilenetv3_large()
model.eval()
# load model weights
if 1:
    model_weight_path = "/media/yiji/ea1def5e-614d-46de-ae59-45f647ac3d2a1/lagopus/PycharmProject/IMAGEclassfier/mobilenetv3.pytorch/logs/20210224_2truckMobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path))
else:
    model_weight_path = "./logs/MobileNetV2.pt"
    model = torch.jit.load(model_weight_path)

def write_pytorch_data(output_path, data, data_name_list):
    """
    Save the data of Pytorch needed to align TNN model.

    The input and output names of pytorch model and onnx model may not match,
    you can use Netron to visualize the onnx model to determine the data_name_list.

    The following example converts ResNet50 to onnx model and saves input and output:
    >>> from torchvision.models.resnet import resnet50
    >>> model = resnet50(pretrained=False).eval()
    >>> input_data = torch.randn(1, 3, 224, 224)
    >>> input_names, output_names = ["input"], ["output"]
    >>> torch.onnx.export(model, input_data, "ResNet50.onnx", input_names=input_names, output_names=output_names)
    >>> with torch.no_grad():
    ...     output_data = model(input_data)
    ...
    >>> write_pytorch_data("input.txt", input_data, input_names)
    >>> write_pytorch_data("output.txt", output_data, output_names)

    :param output_path: Path to save data.
    :param data: The input or output data of Pytorch model.
    :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
    :return:
    """

    if type(data) is not list and type(data) is not tuple:
        data = [data, ]
    assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(data)))
        for name, data in zip(data_name_list, data):
            data = data.numpy()
            shape = data.shape
            description = "{} {} ".format(name, len(shape))
            for dim in shape:
                description += "{} ".format(dim)
            data_type = 0 if data.dtype == np.float32 else 3
            fmt = "%0.6f" if data_type == 0 else "%i"
            description += "{}".format(data_type)
            f.write(description + "\n")
            np.savetxt(f, data.reshape(-1), fmt=fmt)


with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    output_onnx = 'mobilenetv30617.onnx'
    x = img
    print(x.shape)
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    # torch_out = torch.onnx.export(model, img, "super_resolution.onnx", export_params=True, opset_version=10,do_constant_folding=True,
    #                   input_names=['input'], output_names=['output'], dynamic_axes={'input': {0:'batch_size'},'output':{0: 'batch_size'}})
    input_names = ["input0"]
    output_names = ["output0"]
    # torch.onnx.export(model, input_data, "ResNet50.onnx", input_names=input_names, output_names=output_names)
    torch_out = torch.onnx._export(model, x, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)

    # netron.start(output_onnx)
    with torch.no_grad():
        output_data = model(x)

    write_pytorch_data("input.txt", x, input_names)
    write_pytorch_data("output.txt", output_data, output_names)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(output_onnx)


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # print(ort_inputs)
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


    # import onnx
    # onnx_model = onnx.load("super_resolution.onnx")
    # onnx.checker.check_model(onnx_model)

    # predict = torch.softmax(output, dim=0)
    # predict_cla = torch.argmax(predict).numpy()
    # print(predict_cla)
    print("==> Passed")
    # print("==> Loading onnx model into Caffe2 backend and comparing forward pass.".format(output_onnx))
    # caffe2_backend = backend.prepare(onnx_model)
    # B = {"input0": x.data.numpy()}
    # c2_out = caffe2_backend.run(B)["output0"]
    #
    # print("==> compare torch output and caffe2 output")
    # np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
    # print("==> Passed")
    # import onnxruntime
    #
    # ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    #
    #
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
# print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
# plt.show()




