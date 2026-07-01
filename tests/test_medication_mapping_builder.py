import importlib.util
import sys
from pathlib import Path

import duckdb


def load_builder_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_medication_mappings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_medication_mappings", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_builder_module()


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_parquet_rows(
    path: Path,
    columns: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = ", ".join(
        "(" + ", ".join(sql_string(value) for value in row) + ")" for row in rows
    )
    column_sql = ", ".join(columns)
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"CREATE TABLE rows AS SELECT * FROM (VALUES {values}) AS t({column_sql})"
        )
        connection.execute(f"COPY rows TO {sql_string(path)} (FORMAT PARQUET)")


def test_normalizers_are_conservative() -> None:
    assert builder.normalize_ndc("0002-3227-30") == "0002322730"
    assert builder.normalize_ndc("12345678901.0") == "12345678901"
    assert builder.normalize_ndc("ABC-123") == "ABC-123"
    assert (
        builder.normalize_drug_name("Acetaminophen 650 mg PO tablet") == "acetaminophen"
    )
    assert builder.normalize_drug_name("sulfamethoxazole/trimethoprim IV") == (
        "sulfamethoxazole trimethoprim"
    )


def test_build_medication_mappings_with_reference_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    extracts_root = dataset_root / "processed" / "extracts"
    mapping_root = dataset_root / "mappings"
    report_path = tmp_path / "reports" / "medication_mapping_build_report.json"

    write_parquet_rows(
        extracts_root / "mimiciv" / "prescriptions.parquet",
        ("ndc",),
        (
            ("0002-3227-30",),
            ("99999999999",),
            ("",),
        ),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "medication.parquet",
        ("drughiclseqno", "gtc", "drugname"),
        (
            ("123", "GTC1", "Vancomycin 1 GM IV"),
            ("", "GTC2", "Acetaminophen 650 mg PO tablet"),
            ("999", "GTC9", "Unknown med"),
        ),
    )
    reference_root = mapping_root / "medications"
    reference_root.mkdir(parents=True)
    (reference_root / "ndc2RXCUI.txt").write_text(
        "NDC,RXCUI\n0002322730,RX123\n",
        encoding="utf-8",
    )
    (reference_root / "RXCUI2atc4.csv").write_text(
        "\n".join(
            [
                "RXCUI,ATC4,ingredient_name,rxnorm_name",
                "RX123,J01XA,vancomycin,Vancomycin",
                "RX456,N02BE,acetaminophen,Acetaminophen",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (reference_root / "eicu_hicl_rxnorm_atc.csv").write_text(
        "\n".join(
            [
                "drughiclseqno,gtc,drug_name,rxcui,ingredient_name,rxnorm_name,atc_code,atc_level",
                "123,GTC1,Vancomycin,RX123,vancomycin,Vancomycin,J01X,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = builder.build_medication_mappings(
        builder.MedicationMappingBuildConfig(
            dataset_root=dataset_root,
            extracts_root=extracts_root,
            mapping_root=mapping_root,
            report_path=report_path,
        )
    )

    mimic_output = (
        mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv"
    ).read_text(encoding="utf-8")
    eicu_output = (mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv").read_text(
        encoding="utf-8"
    )

    assert report["mimic"]["distinct_ndc_count"] == 2
    assert report["mimic"]["rxcui_mapped_count"] == 1
    assert report["mimic"]["atc3_mapped_count"] == 1
    assert "sample_unmapped_ndcs" not in report["mimic"]
    assert "0002-3227-30,RX123,vancomycin,Vancomycin,J01X,3" in mimic_output
    assert "99999999999,,,,," in mimic_output

    assert report["eicu"]["distinct_eicu_medication_concept_count"] == 3
    assert report["eicu"]["mapped_by_hicl_count"] == 1
    assert report["eicu"]["mapped_by_normalized_name_count"] == 1
    assert "sample_unmapped_concepts" not in report["eicu"]
    assert (
        "123,GTC1,Vancomycin 1 GM IV,RX123,vancomycin,Vancomycin,J01X,3" in eicu_output
    )
    assert (
        ",GTC2,Acetaminophen 650 mg PO tablet,RX456,acetaminophen,Acetaminophen,N02B,3"
        in eicu_output
    )


def test_build_medication_mappings_without_references_keeps_unmapped_rows(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "Dataset"
    extracts_root = dataset_root / "processed" / "extracts"
    mapping_root = dataset_root / "mappings"
    report_path = tmp_path / "reports" / "medication_mapping_build_report.json"

    write_parquet_rows(
        extracts_root / "mimiciv" / "prescriptions.parquet",
        ("ndc",),
        (("11111111111",),),
    )
    write_parquet_rows(
        extracts_root / "eicu_crd" / "medication.parquet",
        ("drughiclseqno", "gtc", "drugname"),
        (("321", "GTC3", "No reference med"),),
    )

    report = builder.build_medication_mappings(
        builder.MedicationMappingBuildConfig(
            dataset_root=dataset_root,
            extracts_root=extracts_root,
            mapping_root=mapping_root,
            report_path=report_path,
        )
    )

    assert report["missing_reference_files"] == [
        "ndc2RXCUI.txt or equivalent",
        "RXCUI2atc4.csv or equivalent",
        "drug-atc.csv or equivalent",
        "reviewed eICU HICL/GTC-to-RxNorm map",
    ]
    assert report["mimic"]["unmapped_count"] == 1
    assert report["eicu"]["unmapped_count"] == 1
    assert (mapping_root / "medications" / "mimic_ndc_rxnorm_atc.csv").exists()
    assert (mapping_root / "medications" / "eicu_drug_rxnorm_atc.csv").exists()
