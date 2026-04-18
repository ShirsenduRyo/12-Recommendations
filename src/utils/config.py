from omegaconf import OmegaConf


def load_config(config_path: str):

    cfg = OmegaConf.load(config_path)

    return cfg


def to_dict(cfg):

    return OmegaConf.to_container(cfg, resolve=True)