import importlib.util
import pytest


def test_smoke_cli_has_explicit_offline_asset_and_input_arguments(capsys):
    assert importlib.util.find_spec('vast_v3.cli') is not None, 'CLI package missing'
    from vast_v3.cli.smoke import main
    with pytest.raises(SystemExit) as result:
        main(['--help'])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ('--dataset', '--raw-root', '--manifest', '--text-directory', '--text-lock', '--output'):
        assert flag in help_text
    assert '--roots' not in help_text


def test_smoke_cli_will_not_use_test_case_as_training_fixture(tmp_path):
    assert importlib.util.find_spec('vast_v3.cli') is not None, 'CLI package missing'
    from vast_v3.cli.smoke import select_smoke_case
    from vast_v3.data.manifests import FrozenSplit
    split = FrozenSplit('rcabench', ('train',), ('test',), 'a' * 64)
    assert select_smoke_case(split, None) == 'train'
    with pytest.raises(ValueError, match='training'):
        select_smoke_case(split, 'test')
