# Copyright (c) 2025 MatN23. All rights reserved.
import copy
import csv
import os
from typing import Dict, List

from colossalai.logging import DistributedLogger

from .base import BaseDataset

cmmlu_subject_mapping = {
    "agronomy": "",
    "anatomy": "",
    "ancient_chinese": "",
    "arts": "",
    "astronomy": "",
    "business_ethics": "",
    "chinese_civil_service_exam": "",
    "chinese_driving_rule": "",
    "chinese_food_culture": "",
    "chinese_foreign_policy": "",
    "chinese_history": "",
    "chinese_literature": "",
    "chinese_teacher_qualification": "",
    "clinical_knowledge": "",
    "college_actuarial_science": "",
    "college_education": "",
    "college_engineering_hydrology": "",
    "college_law": "",
    "college_mathematics": "",
    "college_medical_statistics": "",
    "college_medicine": "",
    "computer_science": "",
    "computer_security": "",
    "conceptual_physics": "",
    "construction_project_management": "",
    "economics": "",
    "education": "",
    "electrical_engineering": "",
    "elementary_chinese": "",
    "elementary_commonsense": "",
    "elementary_information_and_technology": "",
    "elementary_mathematics": "",
    "ethnology": "",
    "food_science": "",
    "genetics": "",
    "global_facts": "",
    "high_school_biology": "",
    "high_school_chemistry": "",
    "high_school_geography": "",
    "high_school_mathematics": "",
    "high_school_physics": "",
    "high_school_politics": "",
    "human_sexuality": "",
    "international_law": "",
    "journalism": "",
    "jurisprudence": "",
    "legal_and_moral_basis": "",
    "logical": "",
    "machine_learning": "",
    "management": "",
    "marketing": "",
    "marxist_theory": "",
    "modern_chinese": "",
    "nutrition": "",
    "philosophy": "",
    "professional_accounting": "",
    "professional_law": "",
    "professional_medicine": "",
    "professional_psychology": "",
    "public_relations": "",
    "security_study": "",
    "sociology": "",
    "sports_science": "",
    "traditional_chinese_medicine": "",
    "virology": "",
    "world_history": "",
    "world_religions": "",
}

default_inference_kwargs = {
    "calculate_loss": True,
    "all_classes": ["A", "B", "C", "D"],
    "language": "Chinese",
    "pretrain": False,
    "max_new_tokens": 32,
}


def get_few_shot_data(data: List[Dict], subject):
    few_shot_data = [f"{subject}"]
    for i in data:
        few_shot_data.append(i["input"] + i["target"])
    return few_shot_data


class CMMLUDataset(BaseDataset):
    """
    Dataset class for CMMLU dataset.
    Data source: https://github.com/haonan-li/CMMLU/tree/master/data
    This dataset class will convert the original dataset into the inference dataset.
    """

    @staticmethod
    def load(
        path: str, logger: DistributedLogger, few_shot: bool, forward_only: bool, load_train: bool, load_reference: bool
    ) -> List[Dict]:
        dataset = {"dev": {}, "test": {}}
        for split in ["dev", "test"]:
            files = os.listdir(os.path.join(path, split))
            files.sort()

            for file in files:
                subject = file[0 : -len(".csv")]
                subject = cmmlu_subject_mapping[subject]

                file_dir = os.path.join(path, split, file)

                dataset[split][subject] = {"data": []}

                # It's been tested that each data sample in one subcategory have same inference arguments.
                dataset[split][subject]["inference_kwargs"] = copy.deepcopy(default_inference_kwargs)

                if split == "test" and few_shot:
                    dataset[split][subject]["inference_kwargs"]["few_shot_data"] = get_few_shot_data(
                        dataset["dev"][subject]["data"], subject
                    )

                with open(file_dir, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    _ = next(reader)
                    for row in reader:
                        assert len(row) == 7
                        choices = f"A. {row[2]}\nB. {row[3]}\nC. {row[4]}\nD. {row[5]}"
                        data_sample = {
                            "dataset": "cmmlu",
                            "split": split,
                            "category": subject,
                            "instruction": f"{subject}",
                            "input": f"{row[1]}\n{choices}\n",
                            "output": "",
                            "target": row[6],
                        }

                        dataset[split][subject]["data"].append(data_sample)

        return dataset