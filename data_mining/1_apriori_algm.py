from itertools import combinations

def load_data():
    return [["面包", "牛奶", "啤酒"],
            ["啤酒", "泡面", "尿布"],
            ["矿泉水", "泡面", "尿布"],
            ["啤酒", "尿布"],
            ["面包", "牛奶", "啤酒", "尿布"],
            ["面包", "牛奶", "啤酒"],
            ["啤酒", "牛奶", "尿布"]]

def get_frequent_itemsets(data, min_support):
    item_count = {}
    for transaction in data:
        for item in transaction:
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
    num_transactions = len(data)
    frequent_itemsets = {frozenset([item]): count / num_transactions for item, count in item_count.items() if count / num_transactions >= min_support}
    k = 2
    while True:
        candidate_itemsets = generate_candidate_itemsets(frequent_itemsets.keys(), k)
        candidate_count = {candidate: 0 for candidate in candidate_itemsets}
        for transaction in data:
            transaction_set = set(transaction)
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction_set):
                    candidate_count[candidate] += 1
        new_frequent_itemsets = {itemset: count / num_transactions for itemset, count in candidate_count.items() if count / num_transactions >= min_support}
        if not new_frequent_itemsets:
            break
        frequent_itemsets.update(new_frequent_itemsets)
        k += 1
    return frequent_itemsets

def generate_candidate_itemsets(itemsets, k):
    candidates = set()
    for itemset1 in itemsets:
        for itemset2 in itemsets:
            union = itemset1 | itemset2
            if len(union) == k and all(frozenset(subset) in itemsets for subset in combinations(union, k-1)):
                candidates.add(union)
    return candidates

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        support_itemset = frequent_itemsets[itemset]
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent_set = frozenset(antecedent)
                consequent_set = itemset - antecedent_set
                confidence = support_itemset / frequent_itemsets[antecedent_set]
                if confidence >= min_confidence:
                    rules.append((antecedent_set, consequent_set, confidence))
    return rules

if __name__ == "__main__":
    data = load_data()
    min_support = 0.3
    min_confidence = 0.8
    frequent_itemsets = get_frequent_itemsets(data, min_support)
    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{list(itemset)}: {support}")
    association_rules = generate_association_rules(frequent_itemsets, min_confidence)
    print("\nAssociation Rules:")
    for antecedent, consequent, confidence in association_rules:
        print(f"{list(antecedent)} -> {list(consequent)}: Confidence = {confidence:.2f}")
