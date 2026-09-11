import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

project_root = Path(".").resolve()
reports_dir = project_root / "reports"
images_dir = project_root / "images"

# Configure figure
fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)
fig.patch.set_facecolor("#0b0f19")

# Load images
img_banner = mpimg.imread(images_dir / "banner_title.jpg")
img_bench = mpimg.imread(reports_dir / "benchmark_tripartite_comparison.png")
img_cm = mpimg.imread(reports_dir / "confusion_matrix_test.png")
img_shap = mpimg.imread(reports_dir / "shap_global_summary.png")

# Plot 1: Banner
axes[0, 0].imshow(img_banner)
axes[0, 0].set_title("1. ODYSSEUS AI PLATFORM - Wine Quality MLOps System", color="#f0e68c", fontsize=14, fontweight="bold", pad=12)
axes[0, 0].axis("off")

# Plot 2: Benchmark
axes[0, 1].imshow(img_bench)
axes[0, 1].set_title("2. Benchmark Empirico Tripartito: Nguyen vs. Rachmaan vs. Odysseus", color="#f0e68c", fontsize=14, fontweight="bold", pad=12)
axes[0, 1].axis("off")

# Plot 3: Confusion Matrix
axes[1, 0].imshow(img_cm)
axes[1, 0].set_title("3. Matriz de Confusion Normalizada (Test Ciego N=1,300, 92%+ exacto o +-1)", color="#f0e68c", fontsize=14, fontweight="bold", pad=12)
axes[1, 0].axis("off")

# Plot 4: SHAP XAI
axes[1, 1].imshow(img_shap)
axes[1, 1].set_title("4. Explicabilidad Causal XAI en Tiempo Real (SHAP TreeExplainer)", color="#f0e68c", fontsize=14, fontweight="bold", pad=12)
axes[1, 1].axis("off")

# Global layout
plt.subplots_adjust(top=0.91, bottom=0.03, left=0.02, right=0.98, hspace=0.18, wspace=0.08)
fig.suptitle(
    "ODYSSEUS AI PLATFORM | Wine Quality Classification & MLOps Framework\nLead Architect: Guillen Concepcion - Senior Data Scientist & MLOps Engineer",
    color="white",
    fontsize=18,
    fontweight="bold",
    y=0.97
)

output_path = reports_dir / "linkedin_infographic_odysseus.png"
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
print(f"SUCCESS: Saved to {output_path} (size: {output_path.stat().st_size} bytes)")
