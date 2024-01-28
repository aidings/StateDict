import json
import torch
from loguru import logger


class StateDict:
    def __init__(self, model:torch.nn.Module, map_key={}, del_key=[]):
        assert isinstance(model, torch.nn.Module), 'model must be torch.nn.Module'
        self.model = model
        self.map_key = map_key
        self.del_key = del_key

    def load(self, ckpt_path, strict=False):
        if isinstance(ckpt_path, dict):
            ckpt = ckpt_path
            for key in ckpt.keys():
                ckpt[key] = ckpt[key].to('cpu')
        elif isinstance(ckpt_path, str) and ckpt_path.endswith('.safetensors'):
            ckpt = self._safe2ckpt(ckpt_path)
        else:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
            except Exception as e:
                raise RuntimeError(str(e))

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # 映射key
        ckpt = self.__map_key(ckpt)
        # 移除多GPU模式的权重
        ckpt = self.__remove_module(ckpt)

        _, match = self.diff(ckpt)

        self.model.load_state_dict(match, strict=strict)
        logger.info(f'load state dict success: {ckpt_path}')

    def diff(self, ckpt):
        info = {"match": 0, "size_not_same": {}, "name_not_same": [], "both_not_same": []}
        match = {}
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            flag = False
            for del_key in self.del_key:
                flag = del_key in key
                if flag:
                   break
            if flag:
                continue
            if key in ckpt.keys():
                # name same
                if state_dict[key].size() == ckpt[key].size():
                    # weight same
                    info['match'] += 1
                    match[key] = ckpt[key]
                else:
                    info['size_not_same'][key] = (list(state_dict[key].size()), list(ckpt[key].size()))
                    info['both_not_same'].append(key)
            else:
                info['name_not_same'].append(key)
                info['both_not_same'].append(key)

        logger.warning(json.dumps({'match': info['match'],
                                   'name_not_same': len(info['name_not_same']),
                                   'size_not_same': len(info['size_not_same']),
                                   'both_not_same': len(info['both_not_same'])},
                                    ensure_ascii=True))
        return info, match


    def __map_key(self, ckpt):
        if len(self.map_key) == 0:
            return ckpt

        kdict = {}
        for key in self.map_key.keys():
            if key in ckpt.keys():
                dst_key = self.map_key[key]
                kdict[dst_key] = ckpt[key]
            else:
                kdict[key] = ckpt[key]
        return kdict

    def __remove_module(self, ckpt):
        kdict = {}
        for key in ckpt.keys():
            if key.startswith('module.'):
                mkey = key[7:]
                kdict[mkey] = ckpt[key]
            else:
                kdict[key] = ckpt[key]
        return kdict

    def _safe2ckpt(self, safe_path):
        from safetensors import safe_open
        ckpt = {}
        with safe_open(safe_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                ckpt[key] = f.get_tensor(key)
        return ckpt
