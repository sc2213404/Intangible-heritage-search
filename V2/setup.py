import timm

model = timm.create_model("hf_hub:timm/inception_v4.tf_in1k", pretrained=True)