import json
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cfd_ml.paths import METRICS_DIR, TRADEOFF_MODEL_DIR
from cfd_ml.tradeoff import (
    TARGET_LABELS,
    TRADEOFF_FEATURES,
    TRADEOFF_TARGETS,
    build_tradeoff_models,
    evaluate_tradeoff_models,
    fit_tradeoff_model,
    load_tradeoff_modeling_data,
    rank_tradeoff_models,
    split_tradeoff_features,
    summarize_tradeoff_scores,
)


def main():
    tradeoff_df = load_tradeoff_modeling_data()
    X, y = split_tradeoff_features(tradeoff_df)
    models = build_tradeoff_models()

    scores, predictions = evaluate_tradeoff_models(models, X, y)
    summary = summarize_tradeoff_scores(scores)
    ranking = rank_tradeoff_models(summary, ranking_target="cl_cd")

    best_model_name = ranking.iloc[0]["model"]
    best_model = fit_tradeoff_model(models[best_model_name], X, y)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TRADEOFF_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = METRICS_DIR / "tradeoff_response_summary.csv"
    scores_path = METRICS_DIR / "tradeoff_response_scores_by_fold.csv"
    predictions_path = METRICS_DIR / "tradeoff_response_predictions.csv"
    ranking_path = METRICS_DIR / "tradeoff_response_ranking.csv"
    model_path = TRADEOFF_MODEL_DIR / "response_model.joblib"
    metadata_path = TRADEOFF_MODEL_DIR / "response_model_metadata.json"

    summary.to_csv(summary_path, index=False)
    scores.to_csv(scores_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    joblib.dump(best_model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "ranking_target": "cl_cd",
                "ranking_target_label": TARGET_LABELS["cl_cd"],
                "features": TRADEOFF_FEATURES,
                "targets": TRADEOFF_TARGETS,
                "training_rows": len(tradeoff_df),
                "cv_splits": 5,
                "random_state": 42,
            },
            indent=2,
        )
        + "\n"
    )

    print("Trade-off response model")
    print(f"  rows: {len(tradeoff_df)}")
    print(f"  features: {', '.join(TRADEOFF_FEATURES)}")
    print(f"  targets: {', '.join(TRADEOFF_TARGETS)}")
    print(f"  best model by CL/CD RMSE: {best_model_name}")
    print()
    print(ranking[["model", "MAE_mean", "RMSE_mean", "R2_mean"]].to_string(index=False))
    print()
    print(summary_path)
    print(scores_path)
    print(predictions_path)
    print(ranking_path)
    print(model_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
