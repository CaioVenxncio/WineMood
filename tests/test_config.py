from pathlib import Path

import pytest

from winemood.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def test_default_config_loads():
    config = load_config()
    assert config.seed == 42
    assert config.labels.scheme == "points_band_v0"
    assert config.preprocessing.keep_negations is False
    assert config.artifacts.dir.is_absolute()
    assert config.artifacts.model_path.name == "model.h5"
    assert DEFAULT_CONFIG_PATH.is_file()


def test_relative_paths_resolve_against_root(tmp_path: Path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[artifacts]\ndir = "models/x"\n', encoding="utf-8")
    config = load_config(cfg, root=tmp_path)
    assert config.artifacts.dir == tmp_path / "models" / "x"


def test_unknown_key_rejected(tmp_path: Path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[split]\ntest_fraction = 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="test_fraction"):
        load_config(cfg)


def test_invalid_split_rejected(tmp_path: Path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[split]\ntest_size = 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg)


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")
