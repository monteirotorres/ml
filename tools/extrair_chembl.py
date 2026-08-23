# -*- coding: utf-8 -*-
"""Extrai atividades IC50 de um alvo do ChEMBL para um CSV bruto.

Usa a API REST do ChEMBL (EBI) diretamente, sem depender do pacote
chembl_webresource_client, para gerar `dados_alvo_bruto.csv` — o arquivo
pré-extraído que o notebook da aula usa como caminho alternativo quando a
API não está acessível em sala.

Alvo padrão: acetilcolinesterase humana (CHEMBL220).

Uso:
    python tools/extrair_chembl.py
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent.parent
ALVO = "CHEMBL220"                      # acetilcolinesterase
API = "https://www.ebi.ac.uk/chembl/api/data"
SAIDA = BASE / "dados_alvo_bruto.csv"

# Campos que o notebook usa na curadoria.
CAMPOS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_relation", "standard_value", "standard_units",
    "pchembl_value", "assay_type", "assay_chembl_id",
    "data_validity_comment", "document_chembl_id", "target_chembl_id",
]


def baixar(url):
    requisicao = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(requisicao, timeout=60) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def extrair():
    registros = []
    offset = 0
    limite = 1000
    while True:
        url = (f"{API}/activity.json?target_chembl_id={ALVO}"
               f"&standard_type=IC50&limit={limite}&offset={offset}")
        dados = baixar(url)
        lote = dados.get("activities", [])
        if not lote:
            break
        for atividade in lote:
            linha = {}
            for campo in CAMPOS:
                linha[campo] = atividade.get(campo)
            registros.append(linha)
        print(f"  offset {offset}: +{len(lote)} (total {len(registros)})")
        proximo = dados.get("page_meta", {}).get("next")
        if not proximo:
            break
        offset += limite
        time.sleep(0.3)
    return registros


def main():
    print(f"Extraindo IC50 de {ALVO} do ChEMBL...")
    registros = extrair()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(SAIDA, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS)
        escritor.writeheader()
        for linha in registros:
            escritor.writerow(linha)
    print(f"\nSalvo: {SAIDA.relative_to(BASE)}  ({len(registros)} linhas)")


if __name__ == "__main__":
    main()
