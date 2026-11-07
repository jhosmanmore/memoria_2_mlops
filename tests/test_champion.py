"""Pruebas unitarias para la lógica de evaluación y promoción de Champion."""
from unittest.mock import MagicMock

import pytest

from src.champion import evaluar_candidato_a_champion


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


def test_no_cumple_umbral_minimo(mock_client):
    metricas = {"r2": 0.75, "rmse": 1200.0}
    cfg_eval = {"metrica_champion": "r2", "umbral_minimo": 0.85, "modo": "max"}

    res = evaluar_candidato_a_champion(mock_client, "modelo-test", metricas, cfg_eval)
    assert res["es_champion"] is False
    assert "No supera el umbral mínimo" in res["motivo"]


def test_primer_modelo_cumple_umbral_promovido(mock_client):
    # Simular que no existe champion previo
    mock_client.get_model_version_by_alias.side_effect = Exception("Alias no encontrado")
    metricas = {"r2": 0.92, "rmse": 600.0}
    cfg_eval = {"metrica_champion": "r2", "umbral_minimo": 0.85, "modo": "max"}

    res = evaluar_candidato_a_champion(mock_client, "modelo-test", metricas, cfg_eval)
    assert res["es_champion"] is True
    assert "Primer modelo registrado" in res["motivo"]


def test_supera_al_champion_actual(mock_client):
    # Simular champion actual con r2 = 0.90
    champion_version = MagicMock()
    champion_version.version = "1"
    champion_version.run_id = "run-champ-1"
    mock_client.get_model_version_by_alias.return_value = champion_version

    champion_run = MagicMock()
    champion_run.data.metrics = {"r2": 0.90}
    mock_client.get_run.return_value = champion_run

    metricas_candidato = {"r2": 0.95}
    cfg_eval = {
        "metrica_champion": "r2",
        "umbral_minimo": 0.85,
        "mejora_minima": 0.01,
        "modo": "max",
    }

    res = evaluar_candidato_a_champion(mock_client, "modelo-test", metricas_candidato, cfg_eval)
    assert res["es_champion"] is True
    assert "Supera al champion v1" in res["motivo"]


def test_no_supera_al_champion_actual(mock_client):
    # Simular champion actual con r2 = 0.95
    champion_version = MagicMock()
    champion_version.version = "1"
    champion_version.run_id = "run-champ-1"
    mock_client.get_model_version_by_alias.return_value = champion_version

    champion_run = MagicMock()
    champion_run.data.metrics = {"r2": 0.95}
    mock_client.get_run.return_value = champion_run

    metricas_candidato = {"r2": 0.93}  # Inferior al champion
    cfg_eval = {
        "metrica_champion": "r2",
        "umbral_minimo": 0.85,
        "mejora_minima": 0.01,
        "modo": "max",
    }

    res = evaluar_candidato_a_champion(mock_client, "modelo-test", metricas_candidato, cfg_eval)
    assert res["es_champion"] is False
    assert "No supera al champion v1" in res["motivo"]
