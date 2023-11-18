Write-Host "This script runs the complete pipeline."

# Set path to python executable
$python = "C:/Program Files/Python38/python.exe"

# Resample meshes and try to get the vertex count between 20% of the target vertex count
& $python ./Rorschach/preprocessing/patch_meshes.py

# Normalize the resampled meshes, e.g. scale them to a unit box and move barycenter to origin
& $python ./Rorschach/preprocessing/preprocess.py

# Visualize the results of the normalization, i.e. showing barycenter offset
& $python ./Rorschach/visualization/visualize_data.py

# Feature extraction of normalized meshes
& $python ./Rorschach/feature_extraction/extraction.py

# Visualize the extracted histogram features in a grid
& $python ./Rorschach/visualization/extraction_visualisation.py

# Gather |c| neighbours for all meshes using the normalized features and normalized data
& $python ./Rorschach/querying/collect_neighbours.py

# Evaluate the neighbours for all meshes and save to overall_results.csv
& $python ./Rorschach/evaluation/evaluate.py

# Plot the evaluation results after gathering neighbours for all meshes
& $python ./Rorschach/evaluation/plot_results.py

# Generate T-SNE plot of the normalized features
& $python ./Rorschach/visualization/dim_reduction.py