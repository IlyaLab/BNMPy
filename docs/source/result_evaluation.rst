BNMPy.result_evaluation
=======================

The result_evaluation module provides tools for evaluating optimization results.

.. automodule:: BNMPy.result_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
-----------

Evaluating Optimization Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import BNMPy

   # Run optimization
   optimizer = BNMPy.ParameterOptimizer(pbn, "experiments.csv")
   result = optimizer.optimize(method='differential_evolution')

   # Evaluate results with plots and report
   evaluator = BNMPy.evaluate_optimization_result(
       result,
       optimizer,
       output_dir="evaluation_results",
       plot_residuals=True,
       save=True,
       detailed=True,
       figsize=(8, 6)
   )

Evaluating a PBN
~~~~~~~~~~~~~~~~

.. code-block:: python

   import BNMPy

   # Evaluate an existing PBN
   pbn = BNMPy.load_pbn_from_file("network.txt")
   exp_data = BNMPy.ExperimentData("experiments.csv")

   results = BNMPy.evaluate_pbn(
       pbn,
       exp_data,
       output_dir="pbn_evaluation",
       config={'steady_state': {'method': 'monte_carlo'}}
   )

   print(f"MSE: {results['mse']:.4f}")
   print(f"Correlation: {results['correlation']:.3f}")

Generated Plots
---------------

The evaluation functions generate several plots to assess model quality:

1. Prediction vs Experimental Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**prediction_vs_experimental.png**

Scatter plot comparing predicted vs experimental values:

- **X-axis**: Experimental values from CSV file
- **Y-axis**: Predicted values from the model
- **Perfect prediction line**: Red dashed line (y=x)
- **Regression line**: Green line showing linear relationship
- **Confidence interval**: Light green shaded area (95% confidence)
- **Statistics**: Correlation coefficient (r), p-value, and MSE

2. Residuals Plot
~~~~~~~~~~~~~~~~~

**residuals.png**

Shows distribution of prediction errors:

- **Left panel**: Residuals vs Predicted values

  - Residuals = Predicted - Experimental
  - Horizontal red line at y=0

- **Right panel**: Histogram of residuals

  - Distribution of prediction errors
  - Shows mean and standard deviation

3. Optimization History Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**optimization_history.png**

Shows MSE progression during optimization:

- **X-axis**: Optimization iterations
- **Y-axis**: Mean Squared Error (MSE)
- **Line**: MSE values over iterations
- **Stagnation periods**: Highlighted if enabled


Output Files
------------

When ``save=True``, the function generates:

- **detailed_results.csv**: Per-experiment predictions and errors
- **evaluation_report.txt**: Summary statistics and model performance
- **prediction_vs_experimental.png**: Prediction quality plot
- **residuals.png**: Residual analysis (if ``plot_residuals=True``)

Example Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   evaluation_results/
   ├── detailed_results.csv
   ├── evaluation_report.txt
   ├── prediction_vs_experimental.png
   └── residuals.png


Evaluation Report
~~~~~~~~~~~~~~~~~

Text file with summary statistics:

.. code-block:: text

   Optimization Evaluation Report
   ==============================
   
   Final MSE: 0.0123
   Correlation: 0.89
   P-value: 1.2e-15
   RMSE: 0.111
   MAE: 0.089
   
   Number of experiments: 10
   Number of measurements: 40
   
   Optimization converged successfully
   Iterations: 245
   Function evaluations: 3675


See Also
--------

- :doc:`parameter_optimizer` - Main optimization interface
- :doc:`simulation_evaluator` - Simulation evaluation
- :doc:`experiment_data` - Experimental data handling