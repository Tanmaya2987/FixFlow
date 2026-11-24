import joblib
import numpy as np
from pathlib import Path

from embedding_service import embedding_service
MODEL_PATH = Path(__file__).resolve().parent / "priority_model.joblib"


class PriorityService:

    def __init__(self):
        self.saved_data = joblib.load(MODEL_PATH)

        # Find the member selected as the winning model.
        winner_name = self.saved_data["winner"]

        winning_member = next(
            (
                member
                for member in self.saved_data["members"]
                if member.get("name") == winner_name
            ),
            None,
        )

        if winning_member is None:
            raise ValueError(
                f"Winning model '{winner_name}' not found in saved artifact."
            )

        # Extract the actual trained classifier.
        self.classifier = winning_member["clf"]

        if not hasattr(self.classifier, "predict"):
            raise TypeError("Loaded classifier does not support predict().")

        print("Loaded model:", winner_name)
        print("Classifier type:", type(self.classifier).__name__)

    def predict_priority(self, ticket_text):

        embedding = embedding_service.generate_embeddings(
            [ticket_text]
        )

        prediction = self.classifier.predict(embedding)[0]

        probabilities = self.classifier.predict_proba(embedding)[0]

        confidence = float(np.max(probabilities))

        return {
            "priority": str(prediction),
            "confidence": round(confidence, 4),
            "needs_review": (
                str(prediction) == "Critical"
                or confidence < self.saved_data["review_threshold"]
            ),
        }


priority_service = PriorityService()