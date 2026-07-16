import json
import subprocess
import sys

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.retrieval.rule_factory import build_in_memory_retrieval_pipeline as _retrieval
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_dictionary_can_load_entries_before_building_merged_indexes() -> None:
    path = "data/dictionaries/seed_concepts.jsonl"

    entries = DictionaryStore.load_entries_jsonl(path)
    store = DictionaryStore.from_jsonl(path)

    assert entries == store.entries
    assert entries[0].concept_id in store.by_concept_id


def test_drug_type_constraint_excludes_icd10() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.retrieve("metformin", EntityType.DRUG)
    assert candidates
    assert all(candidate.code_system == CodeSystem.RXNORM for candidate in candidates)


def test_disease_type_constraint_excludes_rxnorm() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")
    candidates = generator.retrieve("type 2 diabetes", EntityType.DISEASE)
    assert candidates[0].code == "E11"
    assert all(candidate.code_system != CodeSystem.RXNORM for candidate in candidates)


def test_structured_icd_fields_expand_all_names() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    entry = store.by_concept_id["ICD10:E11"]

    assert entry.official_name_vi == "Đái tháo đường type 2"
    assert entry.parent_code == "E10-E14"
    assert "T2DM" in entry.all_names
    assert "đái tháo đường týp II" in entry.all_names


def test_vietnamese_medical_alias_maps_to_icd_code() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    hypertension = generator.retrieve("cao huyết áp", EntityType.DISEASE)
    myocardial_infarction = generator.retrieve("nhồi máu cơ tim", EntityType.DISEASE)
    copd = generator.retrieve("bệnh phổi tắc nghẽn mạn tính", EntityType.DISEASE)
    ckd = generator.retrieve("suy thận mạn", EntityType.DISEASE)
    gerd = generator.retrieve("GERD", EntityType.DISEASE)

    assert hypertension[0].code == "I10"
    assert myocardial_infarction[0].code == "I21.9"
    assert copd[0].code == "J44.9"
    assert ckd[0].code == "N18.9"
    assert gerd[0].code == "K21.9"


def test_dictionary_alias_overlay_expands_runtime_candidate_lookup(tmp_path) -> None:
    alias_path = tmp_path / "aliases.jsonl"
    alias_path.write_text(
        json.dumps(
            {
                "alias": "đái đường",
                "target_concept_id": "ICD10:E11",
                "semantic_type": "DISEASE",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl", alias_overlay_path=alias_path)
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    candidates = generator.retrieve("đái đường", EntityType.DISEASE)

    assert candidates[0].code == "E11"
    assert candidates[0].code_system == CodeSystem.ICD10
    assert all(candidate.code_system != CodeSystem.RXNORM for candidate in candidates)


def test_rxnorm_drug_fields_expand_aliases_without_icd_leakage() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    candidates = generator.retrieve("Ventolin", EntityType.DRUG)

    assert candidates[0].code == "435"
    assert candidates[0].code_system == CodeSystem.RXNORM
    assert all(candidate.code_system != CodeSystem.ICD10 for candidate in candidates)


def test_source_backed_rxnorm_terms_are_dictionary_constrained() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    aspirin = generator.retrieve("ASA", EntityType.DRUG)
    lisinopril = generator.retrieve("Zestril", EntityType.DRUG)
    omeprazole = generator.retrieve("Prilosec", EntityType.DRUG)

    assert aspirin[0].code == "1191"
    assert lisinopril[0].code == "29046"
    assert omeprazole[0].code == "7646"
    assert all(candidate.code_system == CodeSystem.RXNORM for candidate in aspirin)


def test_phase1_frequent_disease_terms_map_to_icd10() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    atrial_fibrillation = generator.retrieve("rung nhĩ", EntityType.DISEASE)
    hyperlipidemia = generator.retrieve("rối loạn lipid máu", EntityType.DISEASE)
    anemia = generator.retrieve("thiếu máu", EntityType.DISEASE)
    infection = generator.retrieve("nhiễm khuẩn", EntityType.DISEASE)
    kidney_stone = generator.retrieve("sỏi thận", EntityType.DISEASE)

    assert atrial_fibrillation[0].code == "I48.91"
    assert hyperlipidemia[0].code == "E78.5"
    assert anemia[0].code == "D64.9"
    assert infection[0].code == "B99.9"
    assert kidney_stone[0].code == "N20.0"
    assert all(candidate.code_system == CodeSystem.ICD10 for candidate in atrial_fibrillation)


def test_phase1_frequent_drug_terms_map_to_rxnorm() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    tylenol = generator.retrieve("Tylenol", EntityType.DRUG)
    lasix = generator.retrieve("Lasix", EntityType.DRUG)
    metoprolol = generator.retrieve("Metoprolol", EntityType.DRUG)
    nitroglycerin = generator.retrieve("NTG", EntityType.DRUG)
    vancomycin = generator.retrieve("Vancomycin", EntityType.DRUG)
    prednisone = generator.retrieve("Prednisone", EntityType.DRUG)
    doxycycline = generator.retrieve("Doxycycline", EntityType.DRUG)

    assert tylenol[0].code == "161"
    assert lasix[0].code == "4603"
    assert metoprolol[0].code == "6918"
    assert nitroglycerin[0].code == "4917"
    assert vancomycin[0].code == "11124"
    assert prednisone[0].code == "8640"
    assert doxycycline[0].code == "3640"
    assert all(candidate.code_system == CodeSystem.RXNORM for candidate in lasix)


def test_phase1_ontology_lite_drug_terms_map_to_rxnorm() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    expected = {
        "gleevec": "282388",
        "allopurinol": "519",
        "cellcept": "68149",
        "torsemide": "38413",
        "glargine": "274783",
        "crestor": "301542",
        "methadone": "6813",
        "plavix": "32968",
        "augmentin": "151392",
        "amiodarone": "703",
        "colace": "82003",
        "bactrim": "10831",
        "heparin": "5224",
        "dilaudid": "3423",
        "toradol": "35827",
        "ertapenem": "325642",
        "nac": "197",
        "laxis": "4603",
        "levofloxacin": "82122",
        "levafloxacin": "82122",
        "advil": "5640",
        "coumadin": "11289",
        "eliquis": "1364430",
        "seroquel": "51272",
        "azathioprine": "1256",
        "prograf": "42316",
        "bumetanide": "1808",
        "percocet": "42844",
        "octreotide": "7617",
        "zosyn": "74170",
        "bicarb": "36676",
        "compazine": "8704",
        "iron": "90176",
        "desmopressin": "3251",
        "vicodin": "214182",
        "cephalexin": "2231",
        "methylprednisolone": "6902",
        "albuterolipratropium": "214199",
        "ipratropium": "7213",
        "diltiazem": "3443",
        "z-pack": "18631",
        "ceftazidime": "2191",
        "prasugrel": "613391",
        "ranexa": "35829",
        "klonopin": "2598",
        "ceftriaxone": "2193",
        "cefepim": "20481",
        "guaifenesin": "5032",
        "taxol": "56946",
        "fulvestrant": "282357",
        "magnesium": "6574",
    }

    for mention, code in expected.items():
        candidates = generator.retrieve(mention, EntityType.DRUG)
        assert candidates[0].code == code
        assert all(candidate.code_system == CodeSystem.RXNORM for candidate in candidates)


def test_phase1_controlled_rxnorm_uses_ingredient_for_bare_brand_and_scd_for_full_product() -> None:
    store = DictionaryStore.from_jsonl(
        "data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl"
    )
    generator = _retrieval(store)

    bare_brand = generator.retrieve("eliquis", EntityType.DRUG)
    full_product = generator.retrieve("apixaban 5 mg oral tablet", EntityType.DRUG)

    assert [(candidate.code, candidate.source) for candidate in bare_brand] == [
        ("1364430", "exact")
    ]
    assert [(candidate.code, candidate.source) for candidate in full_product] == [
        ("1364445", "exact")
    ]


def test_phase1_frequent_symptom_and_lab_terms_are_dictionary_constrained() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    vomiting = generator.retrieve("nôn", EntityType.SYMPTOM)
    abdominal_pain = generator.retrieve("đau bụng", EntityType.SYMPTOM)
    edema = generator.retrieve("phù", EntityType.SYMPTOM)
    potassium = generator.retrieve("kali", EntityType.LAB_TEST)
    wbc = generator.retrieve("bạch cầu", EntityType.LAB_TEST)
    troponin = generator.retrieve("troponin", EntityType.LAB_TEST)

    assert vomiting[0].code == "SYMPTOM_NAUSEA_VOMITING"
    assert abdominal_pain[0].code == "SYMPTOM_ABDOMINAL_PAIN"
    assert edema[0].code == "SYMPTOM_EDEMA"
    assert potassium[0].code == "POTASSIUM"
    assert wbc[0].code == "WBC"
    assert troponin[0].code == "TROPONIN"


def test_phase1_controlled_dictionary_includes_reviewed_vietnamese_symptom_and_lab_aliases() -> None:
    store = DictionaryStore.from_jsonl("data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    hemoptysis = generator.retrieve("ho ra máu", EntityType.SYMPTOM)
    dysphagia = generator.retrieve("khó nuốt", EntityType.SYMPTOM)
    ct = generator.retrieve("chụp CT", EntityType.LAB_TEST)
    urinalysis = generator.retrieve("xét nghiệm nước tiểu", EntityType.LAB_TEST)
    chest_xray = generator.retrieve("chụp x-quang ngực", EntityType.LAB_TEST)
    blood_pressure = generator.retrieve("huyết áp", EntityType.LAB_TEST)
    spo2 = generator.retrieve("SpO2", EntityType.LAB_TEST)
    dysuria = generator.retrieve("đau buốt khi đi tiểu", EntityType.SYMPTOM)
    procedure = generator.retrieve("thủ thuật", EntityType.PROCEDURE)

    assert hemoptysis[0].code == "SYMPTOM_HEMOPTYSIS"
    assert dysphagia[0].code == "SYMPTOM_DYSPHAGIA"
    assert ct[0].code == "CT"
    assert urinalysis[0].code == "URINALYSIS"
    assert chest_xray[0].code == "XRAY"
    assert blood_pressure[0].code == "BLOOD_PRESSURE"
    assert spo2[0].code == "OXYGEN_SATURATION"
    assert dysuria[0].code == "SYMPTOM_DYSURIA"
    assert procedure == []
    assert all(candidate.code_system == CodeSystem.LOCAL for candidate in hemoptysis)
    assert all(candidate.code_system == CodeSystem.LOCAL for candidate in ct)


def test_phase1_controlled_dictionary_includes_reviewed_standard_allowlist() -> None:
    store = DictionaryStore.from_jsonl("data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    expected = {
        "gãy cổ xương đùi": "S72.0",
        "hạ thân nhiệt": "T68",
        "ảo thanh": "R44.0",
    }

    for mention, code in expected.items():
        candidates = generator.retrieve(mention, EntityType.DISEASE)
        assert candidates[0].code == code
        assert candidates[0].code_system == CodeSystem.ICD10


def test_phase1_missed_terms_expand_ner_and_candidate_recall() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    palpitations = generator.retrieve("đánh trống ngực", EntityType.SYMPTOM)
    chest_tightness = generator.retrieve("thắt chặt ngực", EntityType.SYMPTOM)
    syncope = generator.retrieve("ngất", EntityType.SYMPTOM)
    dizziness = generator.retrieve("chóng mặt", EntityType.SYMPTOM)
    chest_discomfort = generator.retrieve("khó chịu vùng ngực", EntityType.SYMPTOM)
    diabetes = generator.retrieve("đái tháo đường", EntityType.DISEASE)
    cancer = generator.retrieve("ung thư", EntityType.DISEASE)
    cardiovascular = generator.retrieve("bệnh tim mạch", EntityType.DISEASE)
    calculus = generator.retrieve("sỏi", EntityType.DISEASE)
    atenolol = generator.retrieve("atenolol", EntityType.DRUG)

    assert palpitations[0].code == "SYMPTOM_PALPITATIONS"
    assert chest_tightness[0].code == "SYMPTOM_CHEST_TIGHTNESS"
    assert syncope[0].code == "SYMPTOM_SYNCOPE"
    assert dizziness[0].code == "SYMPTOM_DIZZINESS"
    assert chest_discomfort[0].code == "SYMPTOM_CHEST_TIGHTNESS"
    assert diabetes[0].code == "E11"
    assert cancer[0].code == "C80.1"
    assert cardiovascular[0].code == "I51.9"
    assert calculus[0].code == "N20.9"
    assert atenolol[0].code == "1202"


def test_phase1_biliary_stone_terms_prefer_specific_icd10_codes() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    gallstone = generator.retrieve("sỏi mật", EntityType.DISEASE)
    bile_duct_stone = generator.retrieve("sỏi ống mật", EntityType.DISEASE)
    distal_bile_duct_stone = generator.retrieve("sỏi đoạn cuối ống mật chủ", EntityType.DISEASE)

    assert gallstone[0].code == "K80.20"
    assert bile_duct_stone[0].code == "K80.50"
    assert distal_bile_duct_stone[0].code == "K80.50"
    assert all(candidate.code_system == CodeSystem.ICD10 for candidate in gallstone)


def test_phase1_ontology_lite_diagnosis_terms_map_to_icd10() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    expected = {
        "hen suyễn": "J45",
        "bệnh trào ngược dạ dày- thực quản không có viêm thực quản": "K21.9",
        "viêm túi mật cấp": "K81.0",
        "viêm túi mật": "K81.9",
        "viêm dạ dày": "K29.7",
        "viêm dạ dày ruột do virus": "A08.4",
        "loét tá tràng": "K26",
        "viêm thực quản": "K20",
        "loét thực quản": "K22.1",
        "ngưng thở khi ngủ": "G47.30",
        "ngưng thở khi ngủ do tắc nghẽn": "G47.33",
        "tăng kali máu": "E87.5",
        "tăng sản tuyến tiền liệt": "N40",
        "bệnh thận đa nang": "Q61.3",
        "viêm bể thận": "N12",
        "viêm bể thận cấp": "N10",
        "viêm phế quản": "J40",
        "phù gai thị": "H47.1",
        "viêm loét đại tràng": "K51.9",
        "bệnh túi thừa": "K57.9",
        "tăng áp động mạch phổi": "I27.20",
        "bệnh mạch máu ngoại biên": "I73.9",
        "thuyên tắc phổi": "I26.99",
        "huyết khối tĩnh mạch sâu": "I82.40",
        "viêm gan virus B": "B18.1",
        "viêm gan virus C": "B18.2",
        "viêm xương tủy": "M86.9",
        "phình động mạch chủ nhỏ": "I71.9",
        "u ác tuyến tiền liệt": "C61",
        "u ác đầu tụy": "C25.0",
    }

    for mention, code in expected.items():
        candidates = generator.retrieve(mention, EntityType.DISEASE)
        assert candidates[0].code == code
        assert all(candidate.code_system == CodeSystem.ICD10 for candidate in candidates)


def test_phase1_ontology_lite_diagnosis_batch2_terms_map_to_icd10() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    expected = {
        "u ác của tuyến tiền liệt": "C61",
        "u ác của đầu tuỵ": "C25.0",
        "bệnh bạch cầu dòng tủy mãn tính": "C92.1",
        "viêm mô tế bào": "L03.90",
        "nhiễm virus Herpes simplex": "B00.9",
        "bệnh thủy đậu": "B01.9",
        "Zona": "B02.9",
        "bệnh phổi kẽ": "J84.9",
        "viêm phổi kẽ": "J84.9",
        "U Sacoit": "D86.85",
        "suy giảm miễn dịch do sử dụng corticoid": "D84.821",
        "u cơ trơn tử cung": "D25.9",
        "ung thư vú trái": "C50.912",
        "ung thư vú di căn": "C50.919",
        "nhiễm Clostridioides difficile": "A04.72",
        "nhiễm trùng huyết": "A41.9",
        "bệnh lý thần kinh ngoại biên": "G62.9",
        "bàng quang thần kinh": "N31.9",
        "Đa u tủy xương": "C90.0",
        "tăng calci máu": "E83.52",
        "hạ kali máu": "E87.6",
        "cường cận giáp nguyên phát": "E21.0",
        "tràn dịch màng phổi": "J90",
        "xẹp phổi": "J98.11",
        "đau thắt ngực không ổn định": "I20.0",
        "bệnh lý chất trắng": "R90.82",
        "xuất huyết dưới nhện": "I60.9",
        "khối máu tụ dưới màng cứng": "I62.00",
        "hạ huyết áp": "I95.9",
        "u ác của đại tràng": "C18.9",
        "xơ vữa động mạch": "I70.90",
    }

    for mention, code in expected.items():
        candidates = generator.retrieve(mention, EntityType.DISEASE)
        assert candidates[0].code == code
        assert all(candidate.code_system == CodeSystem.ICD10 for candidate in candidates)


def test_phase1_empty_file_terms_expand_recall() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    assert generator.retrieve("hẹp động mạch cảnh", EntityType.DISEASE)[0].code == "I65.29"
    assert generator.retrieve("tách thành động mạch chủ", EntityType.DISEASE)[0].code == "I71.00"
    assert generator.retrieve("liệt hai chân", EntityType.DISEASE)[0].code == "G82.20"
    assert generator.retrieve("béo phì", EntityType.DISEASE)[0].code == "E66.9"
    assert generator.retrieve("tiểu tiện không tự chủ", EntityType.DISEASE)[0].code == "R32"
    assert generator.retrieve("sa âm đạo", EntityType.DISEASE)[0].code == "N81.9"
    assert generator.retrieve("bệnh rễ thần kinh", EntityType.DISEASE)[0].code == "M54.10"
    assert generator.retrieve("loét ngón chân", EntityType.DISEASE)[0].code == "L97.509"
    assert generator.retrieve("bàn chân vẹo bẩm sinh", EntityType.DISEASE)[0].code == "Q66.89"
    assert generator.retrieve("gãy xương sườn trái", EntityType.DISEASE)[0].code == "S22.42"
    assert generator.retrieve("vết thương thấu bụng", EntityType.DISEASE)[0].code == "S31.109"
    assert generator.retrieve("giọng khàn", EntityType.SYMPTOM)[0].code == "SYMPTOM_HOARSENESS"
    assert generator.retrieve("tổn thương dây thanh quản", EntityType.DISEASE)[0].code == "J38.3"
    assert generator.retrieve("cơn co tử cung", EntityType.SYMPTOM)[0].code == "SYMPTOM_UTERINE_CONTRACTIONS"


def test_phase1_new_empty_file_terms_expand_recall() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    generator = _retrieval(store, "data/dictionaries/abbreviations.jsonl")

    assert generator.retrieve("xơ gan do rượu", EntityType.DISEASE)[0].code == "K70.3"
    assert generator.retrieve("hội chứng não gan", EntityType.DISEASE)[0].code == "K76.82"
    assert generator.retrieve("xuất huyết nội sọ không do chấn thương", EntityType.DISEASE)[0].code == "I62.9"
    assert generator.retrieve("rối loạn lưỡng cực", EntityType.DISEASE)[0].code == "F31.9"
    assert generator.retrieve("rối loạn lo âu", EntityType.DISEASE)[0].code == "F41.9"
    assert generator.retrieve("rối loạn cảm xúc", EntityType.DISEASE)[0].code == "F39"
    assert generator.retrieve("ý định tự tử", EntityType.DISEASE)[0].code == "R45.851"
    assert generator.retrieve("hoảng sợ", EntityType.DISEASE)[0].code == "F41.0"
    assert generator.retrieve("hoang tưởng", EntityType.DISEASE)[0].code == "F22"
    assert generator.retrieve("clonidine", EntityType.DRUG)[0].code == "2599"
    assert generator.retrieve("suboxone", EntityType.DRUG)[0].code == "352364"


def test_blocked_alias_removes_false_positive_term() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")

    assert store.exact_lookup("hen") == []
    assert store.exact_lookup("hen phế quản")[0].code == "J45"
    assert store.exact_lookup("hen suyễn")[0].code == "J45"
    assert store.exact_lookup("yếu")[0].code == "SYMPTOM_FATIGUE_WEAKNESS"
    assert store.exact_lookup("mệt mỏi")[0].code == "SYMPTOM_FATIGUE_WEAKNESS"


def test_build_dictionaries_validates_vietnamese_alias_table() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_dictionaries.py", "--config", "configs/default.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["concepts"] >= 80
    assert summary["assertion_cues"] >= 70
    assert summary["source_registry_entries"] >= 13
    assert summary["vietnamese_aliases"] >= 68
