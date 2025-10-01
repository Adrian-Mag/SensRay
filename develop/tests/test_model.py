import pytest
from sensray.core.model import PlanetModel


def test_list_and_load_standard_model():
    models = PlanetModel.list_standard_models()
    assert 'prem' in [m.lower() for m in models]

    pm = PlanetModel.from_standard_model('prem')
    assert pm.name.lower() == 'prem'
    assert hasattr(pm, 'nd_file_path')
