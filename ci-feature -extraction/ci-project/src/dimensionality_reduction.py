import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random
import os
from minisom import MiniSom # Import MiniSom library

class GeneticAlgorithmDR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using a Genetic Algorithm.
    """
    def __init__(self, n_features, target_dim, pop_size, n_gen, mut_prob, cx_prob, k_tournament, evaluation_model):
        """
        Initializes the GeneticAlgorithmDR.
        """
        self.n_features = n_features
        self.target_dim = target_dim
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mut_prob = mut_prob
        self.cx_prob = cx_prob
        self.k_tournament = k_tournament
        self.evaluation_model = evaluation_model
        self.best_transformation_matrix = None
        self.best_fitness_history = []

        self._D = self.n_features * self.target_dim

    def _init_population(self):
        return [np.random.randn(self._D) for _ in range(self.pop_size)]

    def _evaluate_individual(self, individual, X_train, X_test, y_train, y_test):
        W = individual.reshape(self.n_features, self.target_dim)
        X_train_proj = X_train @ W
        X_test_proj = X_test @ W
        clf = self.evaluation_model
        try:
            clf_instance = type(clf)(**clf.get_params())
            clf_instance.fit(X_train_proj, y_train)
            preds = clf_instance.predict(X_test_proj)
            return accuracy_score(y_test, preds)
        except Exception as e:
            return -np.inf

    def _tournament_selection(self, population, scores):
        valid_indices = range(len(population))
        k = min(self.k_tournament, len(population))
        selected_indices = np.random.choice(valid_indices, k, replace=False)
        best_in_tournament_idx = selected_indices[np.argmax([scores[j] for j in selected_indices])]
        return population[best_in_tournament_idx]

    def _crossover(self, p1, p2):
        child = p1.copy()
        if np.random.rand() < self.cx_prob:
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
        return child

    def _mutate(self, ind):
        if np.random.rand() < self.mut_prob:
            noise = np.random.randn(*ind.shape) * 0.1
            ind += noise
        return ind

    def fit(self, X_train, X_test, y_train, y_test):
        if X_train is None or X_test is None or y_train is None or y_test is None:
             raise ValueError("Input data for fitting cannot be None.")

        population = self._init_population()
        best_score_overall = -np.inf

        for gen in range(self.n_gen):
            scores = np.array([self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in population])
            current_best_score = np.max(scores)
            self.best_fitness_history.append(current_best_score)

            if current_best_score > best_score_overall:
                best_score_overall = current_best_score
                best_individual_idx = np.argmax(scores)
                self.best_transformation_matrix = population[best_individual_idx].reshape(self.n_features, self.target_dim)

            new_pop = []
            elite_idx = np.argmax(scores)
            new_pop.append(population[elite_idx].copy())

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, scores)
                p2 = self._tournament_selection(population, scores)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop
            print(f"Gen {gen+1}/{self.n_gen}: Best Fitness (Accuracy) = {current_best_score:.4f} (Overall Best: {best_score_overall:.4f})")

        if self.best_transformation_matrix is None:
             print("Warning: GA did not find a valid transformation matrix.")
             self.best_transformation_matrix = np.zeros((self.n_features, self.target_dim))


    def transform(self, X):
        if self.best_transformation_matrix is None:
            raise RuntimeError("Fit method must be called before transform.")
        if X is None:
             raise ValueError("Input data for transform cannot be None.")

        try:
            return X @ self.best_transformation_matrix
        except Exception as e:
            print(f"An error occurred during transformation: {e}")
            return None


class PCADR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Principal Component Analysis (PCA).
    """
    def __init__(self, n_components, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X_train):
        if X_train is None:
             raise ValueError("Input training data for PCA fit cannot be None.")
        try:
            self.pca.fit(X_train)
            print(f"PCA fitted with {self.pca.n_components_} components.")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        except Exception as e:
             raise Exception(f"Error during PCA fit: {e}")


    def transform(self, X):
        if self.pca is None:
            raise RuntimeError("Fit method must be called before transform.")
        if X is None:
             raise ValueError("Input data for PCA transform cannot be None.")
        try:
            return self.pca.transform(X)
        except Exception as e:
             raise Exception(f"Error during PCA transform: {e}")

class DifferentialEvolutionDR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Differential Evolution.
    """
    def __init__(self, n_features, target_dim, pop_size, n_gen, F, CR, evaluation_model, random_state=None):
        """
        Initializes the DifferentialEvolutionDR.
        """
        self.n_features = n_features
        self.target_dim = target_dim
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.F = F
        self.CR = CR
        self.evaluation_model = evaluation_model
        self.random_state = random_state
        self.best_transformation_matrix = None
        self.best_fitness_history = []

        self._D = self.n_features * self.target_dim

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def _init_population(self):
        return [np.random.randn(self._D) for _ in range(self.pop_size)]

    def _evaluate_individual(self, individual, X_train, X_test, y_train, y_test):
        W = individual.reshape(self.n_features, self.target_dim)
        X_train_proj = X_train @ W
        X_test_proj = X_test @ W
        clf = self.evaluation_model
        try:
            clf_instance = type(clf)(**clf.get_params())
            clf_instance.fit(X_train_proj, y_train)
            preds = clf_instance.predict(X_test_proj)
            return accuracy_score(y_test, preds)
        except Exception as e:
            return -np.inf

    def _mutate(self, a, b, c):
        return a + self.F * (b - c)

    def _crossover(self, target, mutant):
        trial = np.copy(target)
        rand_idx = np.random.randint(self._D)
        for i in range(self._D):
            if np.random.rand() < self.CR or i == rand_idx:
                trial[i] = mutant[i]
        return trial


    def fit(self, X_train, X_test, y_train, y_test):
        if X_train is None or X_test is None or y_train is None or y_test is None:
             raise ValueError("Input data for fitting cannot be None.")

        population = self._init_population()
        best_score_overall = -np.inf
        best_individual_overall = None

        current_scores = [self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in population]
        current_best_score = max(current_scores)
        best_idx_init = np.argmax(current_scores)
        best_individual_overall = population[best_idx_init].copy()
        best_score_overall = current_best_score
        self.best_fitness_history.append(current_best_score)


        for gen in range(self.n_gen):
            new_population = []
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                idxs = np.random.choice(indices, 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                target = population[i]

                mutant = self._mutate(a, b, c)
                trial = self._crossover(target, mutant)

                score_trial = self._evaluate_individual(trial, X_train, X_test, y_train, y_test)
                score_target = current_scores[i]

                if score_trial >= score_target:
                    new_population.append(trial)
                    current_scores[i] = score_trial
                else:
                    new_population.append(target)

            population = new_population

            current_best_score_gen = max(current_scores)
            self.best_fitness_history.append(current_best_score_gen)

            if current_best_score_gen > best_score_overall:
                 best_score_overall = current_best_score_gen
                 best_individual_idx = np.argmax(current_scores)
                 best_individual_overall = population[best_individual_idx].copy()


            print(f"Gen {gen+1}/{self.n_gen}: Best Fitness (Accuracy) = {current_best_score_gen:.4f} (Overall Best: {best_score_overall:.4f})")

        self.best_transformation_matrix = best_individual_overall.reshape(self.n_features, self.target_dim)


    def transform(self, X):
        if self.best_transformation_matrix is None:
            raise RuntimeError("Fit method must be called before transform.")
        if X is None:
             raise ValueError("Input data for transform cannot be None.")

        try:
            return X @ self.best_transformation_matrix
        except Exception as e:
            print(f"An error occurred during transformation: {e}")
            return None

class MemeticAlgorithmDR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using a Memetic Algorithm (GA + Local Search).
    """
    def __init__(self, n_features, target_dim, pop_size, n_gen, mut_prob, cx_prob, k_tournament, local_search_prob, evaluation_model, random_state=None):
        """
        Initializes the MemeticAlgorithmDR.
        """
        self.n_features = n_features
        self.target_dim = target_dim
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mut_prob = mut_prob
        self.cx_prob = cx_prob
        self.k_tournament = k_tournament
        self.local_search_prob = local_search_prob
        self.evaluation_model = evaluation_model
        self.random_state = random_state
        self.best_transformation_matrix = None
        self.best_fitness_history = []

        self._D = self.n_features * self.target_dim

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

    def _init_population(self):
        return [np.random.randn(self._D) for _ in range(self.pop_size)]

    def _evaluate_individual(self, individual, X_train, X_test, y_train, y_test):
        W = individual.reshape(self.n_features, self.target_dim)
        X_train_proj = X_train @ W
        X_test_proj = X_test @ W
        clf = self.evaluation_model
        try:
            clf_instance = type(clf)(**clf.get_params())
            clf_instance.fit(X_train_proj, y_train)
            preds = clf_instance.predict(X_test_proj)
            return accuracy_score(y_test, preds)
        except Exception as e:
            return -np.inf

    def _tournament_selection(self, population, scores):
        valid_indices = range(len(population))
        k = min(self.k_tournament, len(population))
        selected_indices = np.random.choice(valid_indices, k, replace=False)
        best_in_tournament_idx = selected_indices[np.argmax([scores[j] for j in selected_indices])]
        return population[best_in_tournament_idx]

    def _crossover(self, p1, p2):
        child = p1.copy()
        if np.random.rand() < self.cx_prob:
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
        return child

    def _mutate(self, ind):
        if np.random.rand() < self.mut_prob:
            noise = np.random.randn(*ind.shape) * 0.1
            ind += noise
        return ind

    def _local_search(self, individual, X_train, X_test, y_train, y_test, iterations=5, step_size=0.05):
        """
        Applies a basic hill-climbing local search to an individual.
        """
        best_ind = individual.copy()
        best_score = self._evaluate_individual(best_ind, X_train, X_test, y_train, y_test)

        for _ in range(iterations):
            candidate = best_ind + np.random.randn(*best_ind.shape) * step_size
            candidate_score = self._evaluate_individual(candidate, X_train, X_test, y_train, y_test)

            if candidate_score > best_score:
                best_ind = candidate
                best_score = candidate_score

        return best_ind

    def fit(self, X_train, X_test, y_train, y_test):
        """
        Runs the Memetic Algorithm to find the best transformation matrix.
        """
        if X_train is None or X_test is None or y_train is None or y_test is None:
             raise ValueError("Input data for fitting cannot be None.")

        population = self._init_population()
        best_score_overall = -np.inf
        best_individual_overall = None

        current_scores = [self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in population]
        current_best_score_gen = max(current_scores)
        best_idx_init = np.argmax(current_scores)
        best_individual_overall = population[best_idx_init].copy()
        best_score_overall = current_best_score_gen
        self.best_fitness_history.append(current_best_score_gen)


        for gen in range(self.n_gen):
            new_pop = []
            gen_best_idx = np.argmax(current_scores)
            new_pop.append(population[gen_best_idx].copy())

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, current_scores)
                p2 = self._tournament_selection(population, current_scores)

                child = self._crossover(p1, p2)
                child = self._mutate(child)

                if np.random.rand() < self.local_search_prob:
                    child = self._local_search(child, X_train, X_test, y_train, y_test)

                new_pop.append(child)

            population = new_pop

            current_scores = [self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in population]
            current_best_score_gen = max(current_scores)
            self.best_fitness_history.append(current_best_score_gen)

            if current_best_score_gen > best_score_overall:
                best_score_overall = current_best_score_gen
                best_individual_idx = np.argmax(current_scores)
                best_individual_overall = population[best_individual_idx].copy()


            print(f"Gen {gen+1}/{self.n_gen}: Best Fitness (Accuracy) = {current_best_score_gen:.4f} (Overall Best: {best_score_overall:.4f})")

        if best_individual_overall is not None:
            self.best_transformation_matrix = best_individual_overall.reshape(self.n_features, self.target_dim)
        else:
             print("Warning: MA did not find a valid transformation matrix.")
             self.best_transformation_matrix = np.zeros((self.n_features, self.target_dim))


    def transform(self, X):
        if self.best_transformation_matrix is None:
            raise RuntimeError("Fit method must be called before transform.")
        if X is None:
             raise ValueError("Input data for transform cannot be None.")

        try:
            return X @ self.best_transformation_matrix
        except Exception as e:
            print(f"An error occurred during transformation: {e}")
            return None

class SOMDR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Self-Organizing Maps (SOM).
    Projects data points to the coordinates of their Best Matching Unit (BMU)
    on a trained SOM grid.
    """
    def __init__(self, grid_dims, sigma=1.0, learning_rate=0.5, num_iterations=1000, random_state=None):
        """
        Initializes the SOMDR.

        Args:
            grid_dims (tuple): A tuple specifying the dimensions of the SOM grid (e.g., (10, 10) for 2D, (5, 5, 5) for 3D).
                               The length of this tuple determines the output dimensionality.
            sigma (float): The radius of the neighborhood function.
            learning_rate (float): The learning rate.
            num_iterations (int): The number of iterations for SOM training.
            random_state (int, optional): Seed for the random number generator.
        """
        if not isinstance(grid_dims, tuple) or len(grid_dims) == 0:
            raise ValueError("grid_dims must be a non-empty tuple specifying grid dimensions.")
        self.grid_dims = grid_dims
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.som = None # MiniSom instance

    def fit(self, X_train, y_train=None): # y_train is not used by SOM fit but kept for compatibility
        """
        Trains the Self-Organizing Map on the training data.

        Args:
            X_train (np.ndarray): Training features (scaled).
            y_train (None): Not used by SOM fit.
        """
        if X_train is None:
             raise ValueError("Input training data for SOM fit cannot be None.")

        # Calculate the total number of neurons in the flattened grid
        total_neurons = np.prod(self.grid_dims)

        # Initialize MiniSom as a 1D array of neurons internally
        self.som = MiniSom(x=int(total_neurons), # MiniSom expects int for x and y
                           y=1,
                           input_len=X_train.shape[1],
                           sigma=self.sigma,
                           learning_rate=self.learning_rate,
                           neighborhood_function='gaussian', # Using gaussian as in your example
                           random_seed=self.random_state)

        # Initialize SOM weights
        self.som.random_weights_init(X_train)

        print(f"Training SOM with grid dimensions {self.grid_dims} ({total_neurons} neurons) for {self.num_iterations} iterations...")
        try:
            self.som.train_random(X_train, num_iteration=self.num_iterations)
            print("SOM training done.")
        except Exception as e:
             raise Exception(f"Error during SOM training: {e}")

    def transform(self, X):
        """
        Projects the input data onto the trained SOM grid.

        Args:
            X (np.ndarray): The input features to transform (scaled).

        Returns:
            np.ndarray: The dimensionality-reduced data (coordinates on the SOM grid).
                        The shape will be (n_samples, len(self.grid_dims)).
        """
        if self.som is None:
            raise RuntimeError("Fit method must be called before transform.")
        if X is None:
             raise ValueError("Input data for transform cannot be None.")

        try:
            # Map each input to its Best Matching Unit (BMU) index in the flattened grid
            bmu_indices = np.array([self.som.winner(x)[0] for x in X]) # MiniSom returns (x, y) for 2D, so [0] gives the 1D index for our setup

            # Map the 1D BMU index back to multi-dimensional grid coordinates
            # This logic needs to be general for any number of grid_dims
            coords = np.zeros((X.shape[0], len(self.grid_dims)), dtype=float)
            flat_index = bmu_indices # Start with the 1D index

            # Calculate coordinates by repeatedly taking modulo and integer division
            # based on grid dimensions in reverse order
            dims = list(self.grid_dims)
            for i in range(len(dims) - 1, -1, -1):
                coords[:, i] = flat_index % dims[i]
                flat_index = flat_index // dims[i]

            return coords.astype(float) # Return coordinates as floats

        except Exception as e:
            print(f"An error occurred during SOM transformation: {e}")
            return None
class HybridGASOMDR(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using a Hybrid Genetic Algorithm and Self-Organizing Map.
    GA optimizes a linear projection to an intermediate space, and SOM is applied
    to this intermediate space for final projection.
    """
    def __init__(self, n_features, intermediate_dim, som_grid_dims,
                 ga_pop_size, ga_n_gen, ga_mut_prob, ga_cx_prob, ga_k_tournament,
                 som_sigma=1.0, som_learning_rate=0.5, som_iterations_eval=500, som_iterations_final=1000,
                 evaluation_model_fitness=None, random_state=None):
        """
        Initializes the HybridGASOMDR.

        Args:
            n_features (int): The number of features in the input data.
            intermediate_dim (int): The desired dimension of the intermediate linear projection space.
            som_grid_dims (tuple): A tuple specifying the dimensions of the SOM grid for the final projection.
                                   The length of this tuple is the final output dimensionality.
            ga_pop_size (int): The number of individuals in the GA population.
            ga_n_gen (int): The number of generations to run the GA.
            ga_mut_prob (float): The probability of mutation for a GA individual (the linear matrix).
            ga_cx_prob (float): The probability of crossover between GA parents.
            ga_k_tournament (int): The number of individuals in each GA tournament selection.
            som_sigma (float): The radius of the SOM neighborhood function.
            som_learning_rate (float): The SOM learning rate.
            som_iterations_eval (int): The number of SOM training iterations *within* the GA evaluation.
            som_iterations_final (int): The number of SOM training iterations for the *final* SOM after GA.
            evaluation_model_fitness: An unfitted model object (e.g., LogisticRegression) used for
                                      evaluating fitness within the GA. Defaults to LogisticRegression.
            random_state (int, optional): Seed for the random number generator.
        """
        if not isinstance(som_grid_dims, tuple) or len(som_grid_dims) == 0:
            raise ValueError("som_grid_dims must be a non-empty tuple specifying SOM grid dimensions.")
        if intermediate_dim <= 0 or intermediate_dim > n_features:
            raise ValueError("intermediate_dim must be between 1 and the number of features.")


        self.n_features = n_features
        self.intermediate_dim = intermediate_dim
        self.som_grid_dims = som_grid_dims
        self.ga_pop_size = ga_pop_size
        self.ga_n_gen = ga_n_gen
        self.ga_mut_prob = ga_mut_prob
        self.ga_cx_prob = ga_cx_prob
        self.ga_k_tournament = ga_k_tournament
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_iterations_eval = som_iterations_eval
        self.som_iterations_final = som_iterations_final
        self.random_state = random_state

        # Default evaluation model for fitness if not provided
        if evaluation_model_fitness is None:
             self.evaluation_model_fitness = LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state)
        else:
             self.evaluation_model_fitness = evaluation_model_fitness


        self.best_transformation_matrix = None # The best linear matrix found by GA
        self.final_som_model = None           # The SOM trained with the best linear projection
        self.best_fitness_history = []

        # The dimension of the GA individuals (flattened linear transformation matrix)
        self._D_ga = self.n_features * self.intermediate_dim

        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Calculate total neurons for the SOM
        self._total_som_neurons = np.prod(self.som_grid_dims)
        if self._total_som_neurons <= 0:
             raise ValueError("Total number of SOM neurons must be greater than 0.")


    def _init_ga_population(self):
        """Initializes the GA population (flattened linear transformation matrices)."""
        return [np.random.random(self._D_ga) for _ in range(self.ga_pop_size)]

    def _project_intermediate(self, individual, X_data):
        """Projects data using the linear transformation matrix from a GA individual."""
        W = individual.reshape(self.n_features, self.intermediate_dim)
        return X_data @ W

    def _som_project_final(self, X_intermediate, som_instance):
        """Projects intermediate data onto a given SOM instance."""
        # Map each input to its Best Matching Unit (BMU) index in the flattened grid
        # som.winner(x) returns (x, y) tuple for MiniSom's internal 2D grid,
        # even if we initialized it as x=total_neurons, y=1. We just need the index.
        bmu_indices = np.array([som_instance.winner(x)[0] for x in X_intermediate])

        # Map the 1D BMU index back to multi-dimensional grid coordinates
        coords = np.zeros((X_intermediate.shape[0], len(self.som_grid_dims)), dtype=float)
        flat_index = bmu_indices
        dims = list(self.som_grid_dims)
        for i in range(len(dims) - 1, -1, -1):
            if dims[i] > 0: # Avoid division by zero
                 coords[:, i] = flat_index % dims[i]
                 flat_index = flat_index // dims[i]
            else:
                 # Handle case where a grid dimension is 0, though validation should prevent this
                 coords[:, i] = 0 # Assign 0 coordinate if dimension is 0

        return coords.astype(float)


    def _evaluate_individual(self, individual, X_train, X_test, y_train, y_test):
        """
        Evaluates the fitness of a single GA individual.
        Fitness is the classifier accuracy on the SOM-projected test data
        obtained from the intermediate projection defined by the individual.
        """
        try:
            # 1. Project training and testing data to intermediate space using the individual's matrix
            X_train_intermediate = self._project_intermediate(individual, X_train)
            X_test_intermediate = self._project_intermediate(individual, X_test)

            # 2. Train a temporary SOM on the intermediate *training* data
            # Instantiate a NEW MiniSom for each individual evaluation
            temp_som = MiniSom(x=int(self._total_som_neurons), # Use total neurons
                               y=1,
                               input_len=self.intermediate_dim, # SOM input length is the intermediate dimension
                               sigma=self.som_sigma,
                               learning_rate=self.som_learning_rate,
                               neighborhood_function='gaussian',
                               random_seed=self.random_state) # Use global random state

            # Initialize SOM weights
            temp_som.random_weights_init(X_train_intermediate)

            # Train the temporary SOM for evaluation iterations
            temp_som.train_random(X_train_intermediate, num_iteration=self.som_iterations_eval)

            # 3. Project intermediate training and testing data onto the trained temporary SOM
            Z_train_final = self._som_project_final(X_train_intermediate, temp_som)
            Z_test_final = self._som_project_final(X_test_intermediate, temp_som)

            # 4. Train and evaluate the fitness classifier on the final SOM-projected data
            # Create a new instance of the fitness evaluation model
            clf_instance = type(self.evaluation_model_fitness)(**self.evaluation_model_fitness.get_params())
            clf_instance.fit(Z_train_final, y_train)
            preds = clf_instance.predict(Z_test_final)
            accuracy = accuracy_score(y_test, preds)

            return accuracy

        except Exception as e:
            # Handle potential errors during projection, SOM training, or classification
            # print(f"Warning: Evaluation failed for individual. Error: {e}") # Keep quiet during GA run
            return -np.inf # Return a very low fitness score

    # --- GA Operators (same as GeneticAlgorithmDR, but operate on the flattened matrix) ---
    def _ga_tournament_selection(self, population, scores):
        valid_indices = range(len(population))
        k = min(self.ga_k_tournament, len(population))
        selected_indices = np.random.choice(valid_indices, k, replace=False)
        best_in_tournament_idx = selected_indices[np.argmax([scores[j] for j in selected_indices])]
        return population[best_in_tournament_idx]

    def _ga_crossover(self, p1, p2):
        child = p1.copy()
        if np.random.rand() < self.ga_cx_prob:
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
        return child

    def _ga_mutate(self, ind):
        if np.random.rand() < self.ga_mut_prob:
            noise = np.random.randn(*ind.shape) * 0.1
            ind += noise
        return ind


    def fit(self, X_train, X_test, y_train, y_test):
        """
        Runs the Hybrid GA+SOM algorithm.
        GA finds the best linear transformation matrix. After GA, a final SOM
        is trained using the best linear projection of the training data.

        Args:
            X_train (np.ndarray): Training features (scaled).
            X_test (np.ndarray): Testing features (scaled).
            y_train (pandas.Series): Training target.
            y_test (pandas.Series): Testing target.
        """
        if X_train is None or X_test is None or y_train is None or y_test is None:
             raise ValueError("Input data for fitting cannot be None.")
        if X_train.shape[1] != self.n_features:
             raise ValueError(f"Input data features ({X_train.shape[1]}) mismatch with n_features ({self.n_features}).")
        if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
             raise ValueError("Mismatch between data and target sample sizes.")


        ga_population = self._init_ga_population()
        best_score_overall = -np.inf
        best_individual_overall = None # The best flattened linear matrix found by GA

        # Evaluate initial population
        current_scores = np.array([self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in ga_population])
        current_best_score = np.max(current_scores)
        best_idx_init = np.argmax(current_scores)
        best_individual_overall = ga_population[best_idx_init].copy()
        best_score_overall = current_best_score
        self.best_fitness_history.append(best_score_overall)

        print(f"Running Hybrid GA+SOM ({self.n_features} -> {self.intermediate_dim} -> {len(self.som_grid_dims)}D SOM Grid {self.som_grid_dims})...")

        for gen in range(self.ga_n_gen):
            new_ga_pop = []
            # Elitism: Keep the best individual from the current generation (or overall best)
            # Add the overall best individual found so far to the new population
            new_ga_pop.append(best_individual_overall.copy())

            # Generate the rest of the new population
            while len(new_ga_pop) < self.ga_pop_size:
                # Select parents using tournament selection based on current scores
                p1 = self._ga_tournament_selection(ga_population, current_scores)
                p2 = self._ga_tournament_selection(ga_population, current_scores)

                # Perform crossover
                child = self._ga_crossover(p1, p2)

                # Perform mutation
                child = self._ga_mutate(child)

                new_ga_pop.append(child)

            ga_population = new_ga_pop

            # Re-evaluate the new population to get scores for the next generation
            current_scores = np.array([self._evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in ga_population])
            current_best_score_gen = np.max(current_scores)
            self.best_fitness_history.append(current_best_score_gen)

            # Update the overall best individual and score
            if current_best_score_gen > best_score_overall:
                best_score_overall = current_best_score_gen
                best_individual_idx = np.argmax(current_scores)
                best_individual_overall = ga_population[best_individual_idx].copy()


            print(f"Gen {gen+1}/{self.ga_n_gen}: Best Fitness (Accuracy) = {current_best_score_gen:.4f} (Overall Best: {best_score_overall:.4f})")

        # After the GA loop finishes, train the final SOM using the best linear matrix found
        if best_individual_overall is not None:
            self.best_transformation_matrix = best_individual_overall.reshape(self.n_features, self.intermediate_dim)

            # Project the *full training data* using the best linear matrix
            X_train_intermediate_final = self._project_intermediate(best_individual_overall, X_train)

            # Train the final SOM
            self.final_som_model = MiniSom(x=int(self._total_som_neurons),
                                          y=1,
                                          input_len=self.intermediate_dim, # SOM input length is intermediate dimension
                                          sigma=self.som_sigma,
                                          learning_rate=self.som_learning_rate,
                                          neighborhood_function='gaussian',
                                          random_seed=self.random_state) # Use global random state

            self.final_som_model.random_weights_init(X_train_intermediate_final)
            print(f"Training final SOM with best GA matrix for {self.som_iterations_final} iterations...")
            self.final_som_model.train_random(X_train_intermediate_final, num_iteration=self.som_iterations_final)
            print("Final SOM training done.")

        else:
             print("Warning: GA did not find a valid transformation matrix. Cannot train final SOM.")
             self.best_transformation_matrix = np.zeros((self.n_features, self.intermediate_dim))
             self.final_som_model = None # No SOM if no valid matrix


    def transform(self, X):
        """
        Applies the learned linear transformation and then the final SOM projection
        to the input data.

        Args:
            X (np.ndarray): The input features to transform (scaled).

        Returns:
            np.ndarray: The dimensionality-reduced data (coordinates on the SOM grid).
                        The shape will be (n_samples, len(self.som_grid_dims)).
        """
        if self.best_transformation_matrix is None or self.final_som_model is None:
            raise RuntimeError("Fit method must be called before transform, and training must have been successful.")
        if X is None:
             raise ValueError("Input data for transform cannot be None.")

        try:
            # 1. Apply the best linear transformation to the input data
            X_intermediate = X @ self.best_transformation_matrix

            # 2. Project the intermediate data onto the final trained SOM
            Z_final_projection = self._som_project_final(X_intermediate, self.final_som_model)

            return Z_final_projection

        except Exception as e:
            print(f"An error occurred during Hybrid GA+SOM transformation: {e}")
            return None

# You would also keep the DataLoader, DataPreprocessor, ModelEvaluator, Visualization classes here
# or import them if they are in separate files in the src directory.