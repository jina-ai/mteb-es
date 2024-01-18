from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval

import datasets


class SpanishPassageRetrievalS2P(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SpanishPassageRetrievalS2P",
            "hf_hub_name": "jinaai/spanish_passage_retrieval",
            "description": "Test collection for passage retrieval from health-related Web resources in Spanish.",
            "reference": "https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(self.description["hf_hub_name"], "queries", split='test', trust_remote_code=True)
        corpus_rows = datasets.load_dataset(self.description["hf_hub_name"], "corpus.documents", split='test', trust_remote_code=True)
        qrels_rows = datasets.load_dataset(self.description["hf_hub_name"], "qrels.s2p", split='test', trust_remote_code=True)

        self.queries = {'test': {row["_id"]: row["text"] for row in query_rows}}
        self.corpus = {'test': {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            'test': {row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows}
        }

        self.data_loaded = True