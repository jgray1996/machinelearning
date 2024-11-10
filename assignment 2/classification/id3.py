import numpy as np

class ID3:
    
    
    def fit(self, data, features, target, depth=0, max_depth=None):
        return self.id3(data, features, target, depth=0, max_depth=None)
        
    def predict(self, df, tree):
        return df.apply(lambda row: self.classify_one(tree, row), axis=1)
    
    def entropy(self, p):
        _, counts = np.unique(p, return_counts=True)
        probs = counts/len(p)
        entropy = -np.sum(probs*np.log2(probs))
        return entropy
    
    def information_gain(self, data, feature, target):
        # Calculate the original entropy of the target variable
        original_entropy = self.entropy(data[target])
        
        # Get the unique values of the feature and their proportions
        feature_values = data[feature].unique()
        total_samples = len(data)
        
        # Weighted entropy after partitioning
        weighted_entropy = 0.0
        
        # For each unique value of the feature
        for value in feature_values:
            # Subset the data for the current value of the feature
            subset = data[data[feature] == value]
            
            # Calculate the entropy of the subset for the target variable
            subset_entropy = self.entropy(subset[target])
            
            # Weight the subset entropy by the proportion of the subset size to the total size
            weighted_entropy += (len(subset) / total_samples) * subset_entropy
        
        # Information gain is the original entropy minus the weighted entropy of the split
        gain = original_entropy - weighted_entropy
        return gain
    
    def id3(self, data, features, target, depth=0, max_depth=None):
        # ID3 algorithm to create a decision tree with max depth
        
        # Select best feature based on information gain to split on (first split should be on Tumor Stage)
        # Stop if the best gain is 0 (no further information can be gained)
        best_feature = None
        max_gain = 0
        for feature in features:
            gain = self.information_gain(data, feature, target)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        
        if max_gain == 0 or (max_depth is not None and depth >= max_depth):
            # No more information gain or maximum depth reached
            # Return the most common target value in the current subset
            return data[target].mode()[0]
        
        # Create a node for the selected feature
        tree = {best_feature: {}}
        
        # Split the data into subsets based on the possible values of the selected feature
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            
            # recursive apply algorithm on each subset, select next best feature until all samples in subset belong to a class or there are no more features
            subtree = self.id3(subset, [feat for feat in features if feat != best_feature], target, depth + 1, max_depth)
            tree[best_feature][value] = subtree
        
        return tree
    

    def classify_one(self, tree, features):
        # recursive tree traverse is inspired by:
        # https://www.geeksforgeeks.org/iterative-dichotomiser-3-id3-algorithm-from-scratch/
        # 
        # Get the first key in the tree
        node = next(iter(tree))
        
        # Get the value of this feature from the features array
        value = features.get(node)
        
        # Get the subtree or classification result based on the feature value
        subtree = tree.get(node, {}).get(value)
        
        # If the subtree is a final classification, return it
        if isinstance(subtree, str):
            return subtree
        
        # Call the function recursively on the subtree
        if isinstance(subtree, dict):
            return self.classify_one(subtree, features)
        
        # If no valid path exists in the tree for the given features, return None or a default value
        return None
