# Copyright (c) 2025 MatN23. All rights reserved.
import copy
import csv
import os
from typing import Dict, List

from colossalai.logging import DistributedLogger

from .base import BaseDataset

ceval_subject_mapping = {
    "computer_network": ["Computer Network", "", "STEM"],
    "operating_system": ["Operating System", "", "STEM"],
    "computer_architecture": ["Computer Architecture", "", "STEM"],
    "college_programming": ["College Programming", "", "STEM"],
    "college_physics": ["College Physics", "", "STEM"],
    "college_chemistry": ["College Chemistry", "", "STEM"],
    "advanced_mathematics": ["Advanced Mathematics", "", "STEM"],
    "probability_and_statistics": ["Probability and Statistics", "", "STEM"],
    "discrete_mathematics": ["Discrete Mathematics", "", "STEM"],
    "electrical_engineer": ["Electrical Engineer", "", "STEM"],
    "metrology_engineer": ["Metrology Engineer", "", "STEM"],
    "high_school_mathematics": ["High School Mathematics", "", "STEM"],
    "high_school_physics": ["High School Physics", "", "STEM"],
    "high_school_chemistry": ["High School Chemistry", "", "STEM"],
    "high_school_biology": ["High School Biology", "", "STEM"],
    "middle_school_mathematics": ["Middle School Mathematics", "", "STEM"],
    "middle_school_biology": ["Middle School Biology", "", "STEM"],
    "middle_school_physics": ["Middle School Physics", "", "STEM"],
    "middle_school_chemistry": ["Middle School Chemistry", "", "STEM"],
    "veterinary_medicine": ["Veterinary Medicine", "", "STEM"],
    "college_economics": ["College Economics", "", "Social Science"],
    "business_administration": ["Business Administration", "", "Social Science"],
    "marxism": ["Marxism", "", "Social Science"],
    "mao_zedong_thought": ["Mao Zedong Thought", "", "Social Science"],
    "education_science": ["Education Science", "", "Social Science"],
    "teacher_qualification": ["Teacher Qualification", "", "Social Science"],
    "high_school_politics": ["High School Politics", "", "Social Science"],
    "high_school_geography": ["High School Geography", "", "Social Science"],
    "middle_school_politics": ["Middle School Politics", "", "Social Science"],
    "middle_school_geography": ["Middle School Geography", "", "Social Science"],
    "modern_chinese_history": ["Modern Chinese History", "", "Humanities"],
    "ideological_and_moral_cultivation": ["Ideological and Moral Cultivation", "", "Humanities"],
    "logic": ["Logic", "", "Humanities"],
    "law": ["Law", "", "Humanities"],
    "chinese_language_and_literature": ["Chinese Language and Literature", "", "Humanities"],
    "art_studies": ["Art Studies", "", "Humanities"],
    "professional_tour_guide": ["Professional Tour Guide", "", "Humanities"],
    "legal_professional": ["Legal Professional", "", "Humanities"],
    "high_school_chinese": ["High School Chinese", "", "Humanities"],
    "high_school_history": ["High School History", "", "Humanities"],
    "middle_school_history": ["Middle School History", "", "Humanities"],
    "civil_servant": ["Civil Servant", "", "Other"],
    "sports_science": ["Sports Science", "", "Other"],
    "plant_protection": ["Plant Protection", "", "Other"],
    "basic_medicine": ["Basic Medicine", "", "Other"],
    "clinical_medicine": ["Clinical Medicine", "", "Other"],
    "urban_and_rural_planner": ["Urban and Rural Planner", "", "Other"],
    "accountant": ["Accountant", "", "Other"],
    "fire_engineer": ["Fire Engineer", "", "Other"],
    "environmental_impact_assessment_engineer": ["Environmental Impact Assessment Engineer", "", "Other"],
    "tax_accountant": ["Tax Accountant", "", "Other"],
    "physician": ["Physician", "", "Other"],
}

default_inference_kwargs = {
    "calculate_loss": False,
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


class CEvalDataset(BaseDataset):
    """
    Dataset class for CEval dataset.
    Data source: https://huggingface.co/datasets/ceval/ceval-exam
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
                subject = file[0 : -len(f"_{split}.csv")]
                subject = ceval_subject_mapping[subject][1]

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
                        # Dev split have answer and explanation so len(row) is 8
                        # But test split doesn't contain answer and explanation, so len(row) is 6
                        assert len(row) >= 6
                        choices = f"A. {row[2]}\nB. {row[3]}\nC. {row[4]}\nD. {row[5]}"
                        data_sample = {
                            "dataset": "ceval",
                            "split": split,
                            "category": subject,
                            "instruction": f"{subject}",
                            "input": f"{row[1]}\n{choices}\n",
                            "output": "",
                            "target": row[6] if split == "dev" else "",
                            "id": int(row[0]),
                        }

                        dataset[split][subject]["data"].append(data_sample)

        return dataset