# Residual MLP Model Comparison

| Model | MAE (m) | RMSE (m) | P95 Abs Error (m) | Max Error (m) |
|-------|---------|----------|-------------------|---------------|
| physics_baseline | 0.0009 | 0.0010 | 0.0016 | 0.0017 |
| linear_regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hist_gradient_boosting | 0.0001 | 0.0002 | 0.0003 | 0.0003 |
| random_forest | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| numpy_mlp_single | 0.0000 | 0.0000 | 0.0001 | 0.0003 |
| numpy_mlp_double | 0.0000 | 0.0000 | 0.0001 | 0.0004 |
