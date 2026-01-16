import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Use relative paths from the repo root
CZOO_ANN = "datasets_cropped_chimpanzee_faces/data_CZoo/annotations_czoo.txt"
CTAI_ANN = "datasets_cropped_chimpanzee_faces/data_CTai/annotations_ctai.txt"

def load_individual_image_counts():
    """
    Parses both CZoo and CTai annotation files and returns a dictionary
    mapping each individual's name to their list of image file paths.
    """
    individuals = defaultdict(list)
    
    def parse_annotation_file(file_path):
        """Parses a single annotation file."""
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Expecting format: Filename <path> Name <name> ...
                if len(parts) >= 4 and parts[0] == 'Filename' and parts[2] == 'Name':
                    image_path = parts[1]
                    name = parts[3]
                    individuals[name].append(image_path)

    parse_annotation_file(CZOO_ANN)
    parse_annotation_file(CTAI_ANN)
    
    return dict(individuals)

def generate_histogram(image_counts, title, save_path):
    """Generates and saves a histogram of image counts."""
    plt.figure(figsize=(12, 7))
    if image_counts:
        max_count = max(image_counts)
        plt.hist(image_counts, bins=range(1, max_count + 2), align='left', rwidth=0.8)
        plt.xticks(range(0, max_count + 1, 5))
    plt.xlabel("Number of Images per Individual")
    plt.ylabel("Number of Individuals")
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.close()

def main(min_images_for_filtered: int = 10):
    """
    Analyzes the chimpanzee dataset, generates statistics and histograms,
    and creates a markdown report.
    """
    all_individuals = load_individual_image_counts()
    
    if not all_individuals:
        print("Error: No individuals found. The annotation files might be empty or in an incorrect format.")
        return

    image_counts_all = [len(images) for images in all_individuals.values()]

    # --- Statistics for All Individuals ---
    total_individuals_all = len(image_counts_all)
    mean_all = np.mean(image_counts_all)
    median_all = np.median(image_counts_all)
    min_all = np.min(image_counts_all)
    max_all = np.max(image_counts_all)

    # --- Generate Histogram for All Individuals ---
    generate_histogram(
        image_counts_all,
        "Images per Individual (All)",
        "images_per_individual_histogram_all.png"
    )

    # --- Filtered Dataset ---
    filtered_individuals = {
        name: images for name, images in all_individuals.items() 
        if len(images) >= min_images_for_filtered
    }
    image_counts_filtered = [len(images) for images in filtered_individuals.values()]
    
    num_kept = len(filtered_individuals)
    num_dropped = total_individuals_all - num_kept

    # --- Statistics for Filtered Individuals ---
    mean_filtered = np.mean(image_counts_filtered) if image_counts_filtered else 0
    median_filtered = np.median(image_counts_filtered) if image_counts_filtered else 0
    min_filtered = np.min(image_counts_filtered) if image_counts_filtered else 0
    max_filtered = np.max(image_counts_filtered) if image_counts_filtered else 0

    # --- Generate Histogram for Filtered Individuals ---
    generate_histogram(
        image_counts_filtered,
        f"Images per Individual (>= {min_images_for_filtered} Images)",
        f"images_per_individual_histogram_min{min_images_for_filtered}.png"
    )

    # --- Identify specific individuals ---
    ids_lt_5 = [name for name, images in all_individuals.items() if len(images) < 5]
    ids_gt_50 = [name for name, images in all_individuals.items() if len(images) > 50]

    # --- Generate Markdown Report ---
    report_content = f"""# Data Analysis Report: Chimpanzee Faces Dataset

This report provides a statistical overview of the Chimpanzee Faces dataset, which is composed of images from two sources: CZoo and CTai. The analysis covers the full dataset and a filtered subset suitable for deep learning tasks.

The dataset is structured as follows:
- `datasets_cropped_chimpanzee_faces/data_CZoo/`: Contains images and annotations for the CZoo collection.
- `datasets_cropped_chimpanzee_faces/data_CTai/`: Contains images and annotations for the CTai collection.

## Statistics — All Individuals

- **Total number of unique individuals:** {total_individuals_all}
- **Image count statistics per individual:**
  - **Mean:** {mean_all:.2f}
  - **Median:** {median_all:.2f}
  - **Min:** {min_all}
  - **Max:** {max_all}

A histogram showing the distribution of images per individual across the entire dataset is saved as `images_per_individual_histogram_all.png`.

![All Individuals Histogram](images_per_individual_histogram_all.png)

## Statistics — Filtered (≥ {min_images_for_filtered} images per individual)

To create a more balanced dataset for training, we filter out individuals with fewer than {min_images_for_filtered} images.

- **Threshold used:** {min_images_for_filtered} images
- **Number of kept individuals:** {num_kept}
- **Number of dropped individuals:** {num_dropped}
- **Image count statistics in the filtered subset:**
  - **Mean:** {mean_filtered:.2f}
  - **Median:** {median_filtered:.2f}
  - **Min:** {min_filtered}
  - **Max:** {max_filtered}

A histogram for this filtered subset is saved as `images_per_individual_histogram_min{min_images_for_filtered}.png`.

![Filtered Individuals Histogram](images_per_individual_histogram_min{min_images_for_filtered}.png)

## Notable Individual Groups

- **Individuals with fewer than 5 images ({len(ids_lt_5)}):**
  - `{", ".join(sorted(ids_lt_5))}`
- **Individuals with more than 50 images ({len(ids_gt_50)}):**
  - `{", ".join(sorted(ids_gt_50))}`

## Suitability for Deep Learning

The filtered dataset (with ≥ {min_images_for_filtered} images per individual) is significantly more suitable for deep learning tasks that rely on recognizing individuals.

- **For classification-based metric learning (e.g., ArcFace, CosFace, SphereFace):** These methods train a classifier to distinguish between identities. Individuals with very few images (e.g., 1-5) provide insufficient data for the model to learn robust, generalizable features for that class, leading to poor convergence and performance. The filtered set ensures each identity has enough examples.

- **For triplet loss:** This method learns an embedding by comparing an "anchor" image to "positive" (same identity) and "negative" (different identity) examples. To form valid triplets, at least two images are required for any given individual. While technically possible with just two images, having a richer set of {min_images_for_filtered} or more images allows for more diverse and effective triplet sampling (e.g., hard-positive and hard-negative mining), which is crucial for training a high-performance model.

Dropping individuals with few samples is a standard practice to improve model stability and ensure that the learned embeddings are based on representative data for each identity.
"""
    with open("Data_Analysis_Report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("Analysis complete. Report and histograms have been regenerated.")
    print(f" - Data_Analysis_Report.md")
    print(f" - images_per_individual_histogram_all.png")
    print(f" - images_per_individual_histogram_min{min_images_for_filtered}.png")


if __name__ == "__main__":
    main()
