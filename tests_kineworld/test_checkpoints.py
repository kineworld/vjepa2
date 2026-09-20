import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.torch=Mock()
        with patch.dict(sys.modules,torch=self.torch):
            spec=importlib.util.spec_from_file_location('backbones_test',Path(__file__).resolve().parents[1]/'src/hub/backbones.py')
            self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def test_default_download_uses_official_https_and_safe_tensor_loading(self):
        self.module._load_pretrained_state('vitl')
        self.torch.hub.load_state_dict_from_url.assert_called_once_with(
            'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt',map_location='cpu',weights_only=True)

    def test_local_checkpoint_does_not_download(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.pt';p.write_bytes(b'fixture')
            self.module._load_pretrained_state('vitl',p)
            self.torch.load.assert_called_once_with(p,map_location='cpu',weights_only=True)
            self.torch.hub.load_state_dict_from_url.assert_not_called()

    def test_missing_local_checkpoint_and_conflicting_option_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):self.module._load_pretrained_state('vitl',Path(d)/'missing')
        for name in ('_make_vjepa2_model','_make_vjepa2_ac_model','_make_vjepa2_1_model'):
            with self.assertRaises(ValueError):getattr(self.module,name)(pretrained=False,checkpoint_path='x')
        self.torch.hub.load_state_dict_from_url.assert_not_called()

if __name__=='__main__':unittest.main()
