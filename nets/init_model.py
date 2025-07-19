from nets import *


def init_model(model_name, args, config):

    if model_name == 's2g_face':
        generator = s2g_face(
            args,
            config,
        )
    elif model_name == 's2g_simple':
        generator = s2g_simple(
            args,
            config,
        )
    elif model_name == 's2g_body_vq':
        generator = s2g_body_vq(
            args,
            config,
        )
    elif model_name == 's2g_body_pixel':
        generator = s2g_body_pixel(
            args,
            config,
        )
    elif model_name == 's2g_body_ae':
        generator = s2g_body_ae(
            args,
            config,
        )
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
            args,
            config,
        )
    elif model_name == 's2g_simple_mlp':
        generator = s2g_simple_mlp(
            args,
            config,
        )
    else:
        raise ValueError
    return generator


