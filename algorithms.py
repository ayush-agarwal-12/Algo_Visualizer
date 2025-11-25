import numpy as np

class MLAlgorithms:
    
    # --- HELPER: DATA GENERATION ---
    @staticmethod
    def generate_data(algo_type, n_samples=200):
        np.random.seed(42)
        
        # 1. Linear & Logistic Data (Blobs)
        if algo_type in ['linear_regression', 'logistic_regression', 'svm']:
            # Class 0
            x0 = np.random.normal(2, 0.6, (n_samples//2, 2))
            y0 = np.zeros((n_samples//2, 1))
            # Class 1
            x1 = np.random.normal(6, 0.6, (n_samples//2, 2))
            y1 = np.ones((n_samples//2, 1))
            
            X = np.vstack((x0, x1))
            y = np.vstack((y0, y1))
            
            if algo_type == 'linear_regression':
                X = 2 * np.random.rand(n_samples, 1)
                y = 4 + 3 * X + np.random.randn(n_samples, 1)
            
            return X, y

        # 2. Non-Linear Data (Parabola)
        elif algo_type == 'polynomial_regression':
            X = 6 * np.random.rand(n_samples, 1) - 3
            y = 0.5 * X**2 + X + 2 + np.random.randn(n_samples, 1)
            return X, y
        
        # 3. Complex Clusters (Moons)
        elif algo_type in ['knn', 'decision_tree', 'dbscan', 'naive_bayes']:
            n_moons = n_samples // 2
            theta = np.linspace(0, np.pi, n_moons)
            r = 1.0 + np.random.normal(0, 0.05, n_moons)
            x1 = r * np.cos(theta)
            y1 = r * np.sin(theta)
            x2 = 1.0 + r * np.cos(theta + np.pi)
            y2 = 0.5 + r * np.sin(theta + np.pi)
            X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
            X = X * 2 
            y = np.hstack((np.zeros(n_moons), np.ones(n_moons))).reshape(-1, 1)
            return X, y

        # 4. Blobs for K-Means
        elif algo_type == 'kmeans':
            c1 = np.random.randn(n_samples//3, 2) + np.array([0, 0])
            c2 = np.random.randn(n_samples//3, 2) + np.array([5, 5])
            c3 = np.random.randn(n_samples//3, 2) + np.array([0, 5])
            X = np.vstack((c1, c2, c3))
            return X, None
            
        return None, None

    # --- HELPER: GRID FOR CONTOURS ---
    @staticmethod
    def make_meshgrid(X, resolution=25):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        return xx, yy, np.c_[xx.ravel(), yy.ravel()]

    
    # 1. REGRESSION

    @staticmethod
    def linear_regression_scratch(learning_rate=0.05, epochs=30):
        X, y = MLAlgorithms.generate_data('linear_regression', 100)
        m, b = np.random.randn(), np.random.randn()
        history = []
        X_b = np.c_[np.ones((len(X), 1)), X]
        theta = np.array([b, m]).reshape(-1,1)
        
        for i in range(epochs):
            gradients = 2/len(X) * X_b.T.dot(X_b.dot(theta) - y)
            theta -= learning_rate * gradients
            line_x = np.linspace(0, 2, 50).reshape(-1,1)
            line_y = np.c_[np.ones((50, 1)), line_x].dot(theta)
            loss = float(np.mean((X_b.dot(theta) - y)**2))
            history.append({
                'step': i, 'line_x': line_x.flatten().tolist(), 'line_y': line_y.flatten().tolist(), 'metric': f"Loss: {loss:.4f}"
            })
        return {'data': {'x': X.flatten().tolist(), 'y': y.flatten().tolist()}, 'frames': history, 'type': 'regression'}

    @staticmethod
    def polynomial_regression_scratch(degree=2, learning_rate=0.0001, epochs=50):
        X, y = MLAlgorithms.generate_data('polynomial_regression', 80)
        X_poly = np.c_[np.ones((len(X), 1)), X, X**2]
        theta = np.random.randn(3, 1)
        history = []
        for i in range(epochs):
            gradients = 2/len(X) * X_poly.T.dot(X_poly.dot(theta) - y)
            theta -= learning_rate * gradients
            line_x = np.linspace(-3, 3, 50).reshape(-1, 1)
            line_y = np.c_[np.ones((50, 1)), line_x, line_x**2].dot(theta)
            loss = float(np.mean((X_poly.dot(theta) - y)**2))
            history.append({
                'step': i, 'line_x': line_x.flatten().tolist(), 'line_y': line_y.flatten().tolist(), 'metric': f"Loss: {loss:.4f}"
            })
        return {'data': {'x': X.flatten().tolist(), 'y': y.flatten().tolist()}, 'frames': history, 'type': 'regression'}

    
    # 2. CLASSIFICATION (LINEAR)
    
    @staticmethod
    def logistic_regression_scratch(learning_rate=0.1, epochs=30):
        X, y = MLAlgorithms.generate_data('logistic_regression', 100)
        X_b = np.c_[np.ones((len(X), 1)), X]
        theta = np.random.randn(3, 1)
        history = []
        for i in range(epochs):
            z = X_b.dot(theta)
            preds = 1 / (1 + np.exp(-z))
            gradient = X_b.T.dot(preds - y) / len(X)
            theta -= learning_rate * gradient
            
            # Decision boundary: w0 + w1x + w2y = 0
            plot_x = np.array([np.min(X[:,0])-1, np.max(X[:,0])+1])
            w2 = theta[2,0] if theta[2,0] != 0 else 1e-5
            plot_y = -(theta[0,0] + theta[1,0] * plot_x) / w2
            acc = np.mean((preds >= 0.5) == y)
            history.append({
                'step': i, 'line_x': plot_x.tolist(), 'line_y': plot_y.tolist(), 'metric': f"Acc: {acc*100:.1f}%"
            })
        return {'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist(), 'c': y.flatten().tolist()}, 'frames': history, 'type': 'classification_line'}

    @staticmethod
    def svm_scratch(learning_rate=0.001, epochs=50, C=1.0):
        # 1. Get Data
        X, y_raw = MLAlgorithms.generate_data('svm', 100)
        y = np.where(y_raw <= 0, -1, 1).flatten()
        
        # 2. Scale Features (Crucial for SVM stability)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_scaled = (X - mean) / std

        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        history = []

        # 3. Training Loop (Soft Margin using Gradient Descent)
        for i in range(epochs):
            for idx, x_i in enumerate(X_scaled):
                condition = y[idx] * (np.dot(x_i, w) - b) >= 1
                if condition:
                    w -= learning_rate * (2 * 1/epochs * w)
                else:
                    w -= learning_rate * (2 * 1/epochs * w - np.dot(x_i, y[idx]) * C)
                    b -= learning_rate * y[idx] * -1 # Gradient of b is -y

            # 4. De-normalize decision boundary for visualization
            # w_scaled * (x - mean)/std - b = 0
            # w_scaled/std * x - (w_scaled*mean/std + b) = 0
            # w_real = w_scaled/std, b_real = b + w_real*mean
            
            w_real = w / std
            b_real = b + np.dot(w_real, mean)

            # Line: w0*x + w1*y - b_real = 0  =>  y = (b_real - w0*x) / w1
            plot_x = np.array([np.min(X[:,0])-1, np.max(X[:,0])+1])
            w1 = w_real[1] if abs(w_real[1]) > 1e-5 else 1e-5
            plot_y = (b_real - w_real[0] * plot_x) / w1

            # Keep plot within reasonable bounds
            plot_y = np.clip(plot_y, -15, 15)

            history.append({
                'step': i,
                'line_x': plot_x.tolist(),
                'line_y': plot_y.tolist(),
                'metric': f"Epoch: {i+1}"
            })

        return {
            'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist(), 'c': y_raw.flatten().tolist()},
            'frames': history,
            'type': 'classification_line'
        }

    
    # 3. CLASSIFICATION (CONTOUR)
    
    @staticmethod
    def knn_scratch(k=3):
        X, y = MLAlgorithms.generate_data('knn', 150)
        xx, yy, grid_points = MLAlgorithms.make_meshgrid(X, resolution=25)
        Z = []
        for point in grid_points:
            distances = np.sqrt(np.sum((X - point)**2, axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = y[nearest_indices].flatten()
            vote = np.bincount(nearest_labels.astype(int)).argmax()
            Z.append(vote)
        Z = np.array(Z).reshape(xx.shape)
        return {
            'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist(), 'c': y.flatten().tolist()},
            'contours': {'z': Z.tolist(), 'x': xx[0].tolist(), 'y': yy[:,0].tolist()},
            'type': 'classification_contour', 'metric': f"KNN (k={k})"
        }

    @staticmethod
    def naive_bayes_scratch():
        X, y = MLAlgorithms.generate_data('naive_bayes', 150)
        classes = np.unique(y)
        summaries = {}
        for c in classes:
            X_c = X[y.flatten() == c]
            summaries[c] = {'mean': X_c.mean(axis=0), 'var': X_c.var(axis=0) + 1e-4}
        def calc_prob(x, mean, var):
            return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-((x - mean)**2 / (2 * var)))
        
        xx, yy, grid_points = MLAlgorithms.make_meshgrid(X, resolution=25)
        Z = []
        for point in grid_points:
            probs = {}
            for c in classes:
                p = np.sum(np.log(calc_prob(point, summaries[c]['mean'], summaries[c]['var'])))
                probs[c] = p
            Z.append(max(probs, key=probs.get))
        Z = np.array(Z).reshape(xx.shape)
        return {
            'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist(), 'c': y.flatten().tolist()},
            'contours': {'z': Z.tolist(), 'x': xx[0].tolist(), 'y': yy[:,0].tolist()},
            'type': 'classification_contour', 'metric': "Gaussian NB"
        }

    @staticmethod
    def decision_tree_scratch(max_depth=3):
        X, y = MLAlgorithms.generate_data('decision_tree', 150)
        y = y.flatten()
        class Node:
            def __init__(self, pred): self.pred = pred; self.idx = 0; self.thr = 0; self.left = None; self.right = None
        
        def build(X_s, y_s, d):
            pred = np.argmax([np.sum(y_s == i) for i in range(2)])
            node = Node(pred)
            if d < max_depth and len(np.unique(y_s)) > 1:
                # Simple random split for visualization effect
                idx = np.random.randint(0, 2)
                thr = np.mean(X_s[:, idx])
                left, right = X_s[:, idx] < thr, X_s[:, idx] >= thr
                if np.sum(left) and np.sum(right):
                    node.idx, node.thr = idx, thr
                    node.left = build(X_s[left], y_s[left], d + 1)
                    node.right = build(X_s[right], y_s[right], d + 1)
            return node
        
        def predict(node, s):
            if node.left: return predict(node.left if s[node.idx] < node.thr else node.right, s)
            return node.pred

        tree = build(X, y, 0)
        xx, yy, gp = MLAlgorithms.make_meshgrid(X, resolution=25)
        Z = np.array([predict(tree, p) for p in gp]).reshape(xx.shape)
        return {
            'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist(), 'c': y.tolist()},
            'contours': {'z': Z.tolist(), 'x': xx[0].tolist(), 'y': yy[:,0].tolist()},
            'type': 'classification_contour', 'metric': f"Tree (Depth {max_depth})"
        }

    
    # 4. CLUSTERING
    
    @staticmethod
    def kmeans_scratch(k=3, max_iters=10):
        X, _ = MLAlgorithms.generate_data('kmeans', 150)
        centroids = X[np.random.choice(len(X), k, replace=False)]
        history = []
        for i in range(max_iters):
            dists = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(dists, axis=0)
            history.append({'step': i, 'centroids': centroids.tolist(), 'labels': labels.tolist(), 'metric': f"Iter: {i+1}"})
            new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels==j)>0 else centroids[j] for j in range(k)])
            if np.all(centroids == new_centroids): break
            centroids = new_centroids
        return {'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist()}, 'frames': history, 'type': 'clustering_kmeans'}

    @staticmethod
    def dbscan_scratch(epsilon=0.5, min_pts=3):
        X, _ = MLAlgorithms.generate_data('dbscan', 150)
        labels = -1 * np.ones(len(X))
        visited = np.zeros(len(X), dtype=bool)
        cluster_id = 0
        history = []
        
        def get_neighbors(i): return np.where(np.linalg.norm(X - X[i], axis=1) < epsilon)[0]
        
        for i in range(len(X)):
            if visited[i]: continue
            visited[i] = True
            neighbors = get_neighbors(i)
            if len(neighbors) < min_pts: labels[i] = -1
            else:
                cluster_id += 1
                labels[i] = cluster_id
                seed = list(neighbors)
                k = 0
                while k < len(seed):
                    curr = seed[k]; k += 1
                    if not visited[curr]:
                        visited[curr] = True
                        new_n = get_neighbors(curr)
                        if len(new_n) >= min_pts: seed.extend([n for n in new_n if n not in seed])
                    if labels[curr] == -1: labels[curr] = cluster_id
            
            if i % 5 == 0 or i == len(X)-1:
                history.append({'step': i, 'labels': labels.tolist(), 'metric': f"Clusters: {cluster_id}"})
                
        return {'data': {'x': X[:,0].tolist(), 'y': X[:,1].tolist()}, 'frames': history, 'type': 'clustering_dbscan'}