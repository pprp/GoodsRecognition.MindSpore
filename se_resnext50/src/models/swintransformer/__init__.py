from .swintransformer import SwinTransformer

def get_swintransformer(num_classes):
    """get swintransformer according to args"""
    # override args
    image_size = 224
    patch_size = 4
    in_chans = 3
    embed_dim = 96
    depths = [ 2, 2, 6, 2 ]
    num_heads = [ 3, 6, 12, 24 ]
    window_size = 7
    drop_path_rate = 0.2
    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    ape = False
    patch_norm = True 
    print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    print(f"==> IMAGE_SIZE:         {image_size}")
    print(f"==> PATCH_SIZE:         {patch_size}")
    print(f"==> NUM_CLASSES:        {num_classes}")
    print(f"==> EMBED_DIM:          {embed_dim}")
    print(f"==> NUM_HEADS:          {num_heads}")
    print(f"==> DEPTHS:             {depths}")
    print(f"==> WINDOW_SIZE:        {window_size}")
    print(f"==> MLP_RATIO:          {mlp_ratio}")
    print(f"==> QKV_BIAS:           {qkv_bias}")
    print(f"==> QK_SCALE:           {qk_scale}")
    print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    print(f"==> APE:                {ape}")
    print(f"==> PATCH_NORM:         {patch_norm}")
    print(25 * "=" + "FINISHED" + 25 * "=")
    model = SwinTransformer(image_size=image_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            num_classes=num_classes,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=0.,
                            drop_path_rate=drop_path_rate,
                            ape=ape,
                            patch_norm=patch_norm)
    # print(model)
    return model


def swin_tiny_patch4_window7_224(num_classes=2388):
    """swin_tiny_patch4_window7_224"""
    return get_swintransformer(num_classes=num_classes)