import pandas as pd
 
from Data import leave_one_out_split, load_ratings
from Metrics import evaluate
 
 
class PopularityRecommender:
    def __init__(self, train: pd.DataFrame):
        # popularity in train
        self.ranking = train["item"].value_counts().index.tolist()
        self.seen = train.groupby("user")["item"].apply(set).to_dict()
 
    def recommend(self, user, k=10):
        seen = self.seen.get(user, set())
        result = []
        for item in self.ranking:
            if item not in seen:
                result.append(item)
                if len(result) == k:
                    break
        return result
 
 
if __name__ == "__main__":
    ratings = load_ratings()
    train, test = leave_one_out_split(ratings)
    test_relevant = test.groupby("user")["item"].apply(set).to_dict()
 
    model = PopularityRecommender(train)
    for k in (10, 20):
        scores = evaluate(model.recommend, test_relevant, k=k)
        print(f"K={k}: " + "  ".join(f"{n}={v:.4f}" for n, v in scores.items()))