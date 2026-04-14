from importlib import import_module

__all__ = ["MODEL_REGISTRY", "AVAILABLE_MODELS", "get_model"]

MODEL_REGISTRY = {
    'MobileNetV2_dynamicFPNPixel2': ('.segmentation.mobilenet_v2_fpn_pixel2', 'MobileNetV2_dynamicFPNPixel2'),
    'MobileNetV2_dynamicFPN': ('.segmentation.mobileunet_v2_crm', 'MobileNetV2_dynamicFPN'),
    'MobileNetV2_unet': ('.segmentation.mobileunet_v2', 'MobileNetV2_unet'),
    'MobileViTv3_v1_dynamicFPN_DANN': ('.segmentation.mobilevit_v3_dann', 'MobileViTv3_v1_dynamicFPN_DANN'),
    'MobileViTv3_v1_dynamicFPN2': ('.segmentation.mobilevit_v3_fpn_plus', 'MobileViTv3_v1_dynamicFPN2'),
    'MobileViTv3_v1_dynamicFPN_LSA': ('.segmentation.mobilevit_v3_lsa', 'MobileViTv3_v1_dynamicFPN_LSA'),
    'MobileViTv3_v1_dynamicFPNpixel2': ('.segmentation.mobilevit_v3_pixel2', 'MobileViTv3_v1_dynamicFPNpixel2'),
    'MobileViTv3_v1_dynamicFPNpixel': ('.segmentation.mobilevit_v3_pixelv1', 'MobileViTv3_v1_dynamicFPNpixel'),
    'MobileViTv3_v1_dynamicFPN_plus': ('.segmentation.mobilevit_v3_plus', 'MobileViTv3_v1_dynamicFPN_plus'),
    'MobileViTv3_v1_SegFormer': ('.segmentation.mobilevit_v3_segformer', 'MobileViTv3_v1_SegFormer'),
    'MobileViTv3_v1_dynamicFPN_SPT': ('.segmentation.mobilevit_v3_spt', 'MobileViTv3_v1_dynamicFPN_SPT'),
    'MobileViTv3_v1_UNET': ('.segmentation.mobilevit_v3_unet', 'MobileViTv3_v1_UNET'),
    'MobileViTv3Lite_v1_dynamicFPN': ('.transformers.lite_mobilevit.lite_v1', 'MobileViTv3Lite_v1_dynamicFPN'),
    'MobileViTv3Litev2_v1_dynamicFPN': ('.transformers.lite_mobilevit.lite_v2', 'MobileViTv3Litev2_v1_dynamicFPN'),
    'MobileViTv3Litev3_v1_dynamicFPN': ('.transformers.lite_mobilevit.lite_v3', 'MobileViTv3Litev3_v1_dynamicFPN'),
    'MobileViTv3Litev4_v1_dynamicFPN': ('.transformers.lite_mobilevit.lite_v4', 'MobileViTv3Litev4_v1_dynamicFPN'),
}

# Backward-compatible alias used by training/evaluation scripts.
AVAILABLE_MODELS = MODEL_REGISTRY

def get_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}')
    module_path, class_name = MODEL_REGISTRY[model_name]
    module = import_module(module_path, package=__name__)
    model_class = getattr(module, class_name)
    return model_class(**kwargs)
