from datetime import datetime, timedelta, timezone

import numpy as np

from app.ml.preparacao import preparar
from app.schemas import Leitura

T0 = datetime(2026, 10, 1, 12, 0)


def test_ordena_e_remove_leituras_duplicadas():
    leituras = [
        Leitura(data_leitura=T0 + timedelta(minutes=20), frequencia_cardiaca=90),
        Leitura(data_leitura=T0, frequencia_cardiaca=80),
        Leitura(data_leitura=T0, frequencia_cardiaca=85),  # reenvio do mesmo instante
    ]
    serie = preparar(leituras)
    assert len(serie) == 2
    assert serie.datas == [T0, T0 + timedelta(minutes=20)]
    assert serie.valores["frequencia_cardiaca"].tolist() == [85, 90]


def test_descarta_valores_impossiveis_do_sensor():
    leituras = [
        Leitura(data_leitura=T0, frequencia_cardiaca=999, temperatura=38.5),
        Leitura(data_leitura=T0 + timedelta(minutes=10), frequencia_cardiaca=0, temperatura=55),
        Leitura(data_leitura=T0 + timedelta(minutes=20), frequencia_cardiaca=95, temperatura=38.6),
    ]
    serie = preparar(leituras)
    assert serie.descartados == 3
    assert np.isnan(serie.valores["frequencia_cardiaca"][0])
    # a segunda leitura não tem nenhum sinal válido e é removida
    assert len(serie) == 2


def test_converte_pressao_em_cmhg_para_mmhg():
    serie = preparar([
        Leitura(data_leitura=T0, pressao=13),
        Leitura(data_leitura=T0 + timedelta(minutes=10), pressao=135),
    ])
    assert serie.valores["pressao"].tolist() == [130, 135]


def test_datas_com_fuso_viram_utc_sem_fuso():
    brasilia = timezone(timedelta(hours=-3))
    serie = preparar([Leitura(data_leitura=datetime(2026, 10, 1, 9, 0, tzinfo=brasilia), frequencia_cardiaca=80)])
    assert serie.datas[0] == datetime(2026, 10, 1, 12, 0)
    assert serie.datas[0].tzinfo is None
