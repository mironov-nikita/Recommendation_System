# Metrics
import math
 
 
def hit_rate_at_k(ranked, relevant, k):
    return float(any(item in relevant for item in ranked[:k]))
 
 
def precision_at_k(ranked, relevant, k):
    hits = sum(item in relevant for item in ranked[:k])
    return hits / k
 
 
def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(item in relevant for item in ranked[:k])
    return hits / len(relevant)
 
 
def ndcg_at_k(ranked, relevant, k):
    dcg = sum(
        1.0 / math.log2(pos + 2)  
        for pos, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
 
 
def evaluate(recommend_fn, test_relevant, k=10):
    """
    test_relevant: dict {user: set(items)}
    """
    sums = {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    users = list(test_relevant)
    for user in users:
        ranked = recommend_fn(user, k)
        rel = test_relevant[user]
        sums["hit_rate"] += hit_rate_at_k(ranked, rel, k)
        sums["precision"] += precision_at_k(ranked, rel, k)
        sums["recall"] += recall_at_k(ranked, rel, k)
        sums["ndcg"] += ndcg_at_k(ranked, rel, k)
    return {name: value / len(users) for name, value in sums.items()}