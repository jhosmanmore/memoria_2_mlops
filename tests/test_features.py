"""Pruebas unitarias para la selección de características (configs/features.yaml)."""
import pytest

from src.features import obtener_features


def test_seleccion_all():
    cfg = {
        "seleccion": "ALL",
        "todas": {
            "categoricas": ["marca", "tipo_vehiculo"],
            "numericas": ["año_modelo", "kilometraje", "precio_lista"],
        },
    }
    info = obtener_features(cfg)
    assert info["alias"] == "ALL"
    assert info["categoricas"] == ["marca", "tipo_vehiculo"]
    assert info["numericas"] == ["año_modelo", "kilometraje", "precio_lista"]
    assert info["n_features"] == 5
    assert len(info["todas"]) == 5


def test_seleccion_numeric():
    cfg = {
        "seleccion": "NUMERIC",
        "todas": {
            "categoricas": ["marca", "tipo_vehiculo"],
            "numericas": ["año_modelo", "kilometraje", "precio_lista"],
        },
    }
    info = obtener_features(cfg)
    assert info["alias"] == "NUMERIC"
    assert info["categoricas"] == []
    assert info["numericas"] == ["año_modelo", "kilometraje", "precio_lista"]
    assert info["n_features"] == 3


def test_seleccion_custom():
    cfg = {
        "seleccion": "CUSTOM",
        "custom": {
            "categoricas": ["marca"],
            "numericas": ["año_modelo", "precio_lista"],
        },
    }
    info = obtener_features(cfg)
    assert info["alias"] == "CUSTOM"
    assert info["categoricas"] == ["marca"]
    assert info["numericas"] == ["año_modelo", "precio_lista"]
    assert info["n_features"] == 3


def test_seleccion_invalida():
    cfg = {"seleccion": "INEXISTENTE"}
    with pytest.raises(ValueError, match="no soportada"):
        obtener_features(cfg)


def test_seleccion_vacia_lanza_error():
    cfg = {"seleccion": "CUSTOM", "custom": {"categoricas": [], "numericas": []}}
    with pytest.raises(ValueError, match="no produjo ninguna característica"):
        obtener_features(cfg)
