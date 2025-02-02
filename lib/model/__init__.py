import importlib


def create_trainer(opt):
    Trainer = importlib.import_module(opt.model.trainer_module).V2AModel(opt)
    return Trainer


def create_model(opt, *args, **kwargs):
    model = importlib.import_module(opt.network_module).V2A(opt, *args, **kwargs)
    return model


def load_loss(opt):
    model = importlib.import_module(opt.module).Loss(opt)
    return model


def load_deformer(opt, rest_pose):
    model = importlib.import_module(opt.module).Deformer(opt, rest_pose)
    return model


def load_sdf_net(opt):
    model = importlib.import_module(opt.module).SDFNet(opt)
    return model


def load_rgb_net(opt):
    model = importlib.import_module(opt.module).RGBNet(opt)
    return model