from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class FEVER(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "FEVER",
            "beir_name": "fever",
            "description": (
                "FEVER (Fact Extraction and VERification) consists of 185,445 claims generated by altering sentences"
                " extracted from Wikipedia and subsequently verified without knowledge of the sentence they were"
                " derived from."
            ),
            "reference": "https://fever.ai/",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["dev", "test"],
            "eval_langs": ["en"],
            "main_score": "map",
        }
