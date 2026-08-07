from clingrounder.benchmarks.phase1.disease_symptom_verifier import (
    DiseaseSymptomLabel,
    DiseaseSymptomVerifier,
    DiseaseSymptomVerifierDataset,
    DiseaseSymptomVerifierExample,
    extract_disease_symptom_features,
    fit_disease_symptom_verifier,
)
from clingrounder.evaluation.sparse_logistic import SparseLogisticModel


def _constant_model(bias: float) -> SparseLogisticModel:
    return SparseLogisticModel(feature_names=(), weights=(), bias=bias)


def _example(
    document_id: str,
    split: str,
    label: DiseaseSymptomLabel,
    feature: str,
) -> DiseaseSymptomVerifierExample:
    return DiseaseSymptomVerifierExample(
        document_id=document_id,
        split=split,
        text=feature,
        position=(0, len(feature)),
        label=label,
        reason="fixture",
        representation_labels=(),
        features=((feature, 1.0),),
    )


def test_verifier_never_uses_disease_as_fallback() -> None:
    verifier = DiseaseSymptomVerifier(
        models=(
            (DiseaseSymptomLabel.DISEASE, _constant_model(1.0)),
            (DiseaseSymptomLabel.NONE, _constant_model(-5.0)),
            (DiseaseSymptomLabel.SYMPTOM, _constant_model(0.99)),
        ),
        disease_threshold=0.0,
        symptom_threshold=0.0,
        minimum_margin=0.10,
        training_dataset_sha256="a" * 64,
    )

    assert verifier.predict({}) is DiseaseSymptomLabel.NONE
    restored = DiseaseSymptomVerifier.from_dict(verifier.to_dict())
    assert restored.predict({}) is DiseaseSymptomLabel.NONE
    assert restored.to_dict()["operating_point"]["disease_fallback"] is False


def test_features_include_section_question_role_and_representation() -> None:
    text = "Hỏi: đau ngực là gì?\nTrả lời: đau ngực là một triệu chứng."
    start = text.rindex("đau ngực")
    end = start + len("đau ngực")

    features = extract_disease_symptom_features(
        text,
        (start, end),
        representation_labels=("DISEASESYMTOM",),
    )

    assert features["qa_role:answer"] == 1.0
    assert features["genre:question_answer"] == 1.0
    assert any(name.startswith("hash:representation_label:") for name in features)


def test_fit_three_way_verifier_reports_target_macro_f1() -> None:
    examples = (
        _example("1", "train", DiseaseSymptomLabel.DISEASE, "disease"),
        _example("2", "train", DiseaseSymptomLabel.DISEASE, "diagnosis"),
        _example("3", "train", DiseaseSymptomLabel.SYMPTOM, "symptom"),
        _example("4", "train", DiseaseSymptomLabel.SYMPTOM, "complaint"),
        _example("5", "train", DiseaseSymptomLabel.NONE, "none"),
        _example("6", "train", DiseaseSymptomLabel.NONE, "drug"),
        _example("7", "development", DiseaseSymptomLabel.DISEASE, "disease"),
        _example("8", "development", DiseaseSymptomLabel.SYMPTOM, "symptom"),
        _example("9", "development", DiseaseSymptomLabel.NONE, "none"),
    )
    dataset = DiseaseSymptomVerifierDataset(
        examples=examples,
        manifest={
            "feature_contract": "phase1-disease-symptom-features.v1",
            "examples_sha256": "b" * 64,
        },
    )

    verifier, report = fit_disease_symptom_verifier(dataset)

    assert verifier.training_dataset_sha256 == "b" * 64
    assert report["holdout_opened"] is False
    assert 0.0 <= report["metrics"]["development"]["target_macro_f1"] <= 1.0
    assert report["policy"]["disease_fallback"] is False
