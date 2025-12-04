import random
from collections import Counter


class CreativeGenerator:
    """
    Suggests new creative message variations for low-CTR campaigns.
    Uses dataset phrases as building blocks.
    """

    def __init__(self, df):
        self.df = df.copy()

    def _extract_phrases(self, texts, top_k=20):
        words = []
        for t in texts:
            if not isinstance(t, str):
                continue
            for w in t.lower().split():
                clean = w.strip(".,!?:;")
                if len(clean) > 2:
                    words.append(clean)
        common = Counter(words).most_common(top_k)
        return [w for w, _ in common]

    def generate_for_campaigns(self, campaigns, n=5):
        output = {}

        for campaign in campaigns:
            rows = self.df[self.df["campaign_name"] == campaign]
            creatives = rows["creative_message"].dropna().astype(str).tolist()

            if not creatives:
                # fallback messages
                output[campaign] = {
                    "suggestions": [
                        "Try a benefit-first headline.",
                        "Add clear CTA such as 'Shop Now'.",
                        "Highlight discount or free shipping.",
                    ],
                    "source_examples": []
                }
                continue

            phrases = self._extract_phrases(creatives)

            suggestions = []
            for _ in range(n):
                if len(phrases) >= 3:
                    headline = " ".join(random.sample(phrases[:10], 3)).title()
                else:
                    headline = "Discover Comfort Today"

                cta = random.choice(["Shop Now", "Buy Today", "Limited Offer", "Get Yours", "Explore"])

                suggestions.append(
                    f"{headline}. {cta}. Highlight key benefits like comfort, material, or value."
                )

            output[campaign] = {
                "suggestions": suggestions,
                "source_examples": creatives[:5]
            }

        return output
