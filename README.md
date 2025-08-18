# Multibe-Linear-Regression
Multiple Linear Regression Pipeline (scikit‑learn)
End‑to‑end ML pipeline for multiple linear regression with data cleaning, outlier handling, log transforms, multicollinearity control (VIF), scaling, one‑hot encoding, train/test split, evaluation (R²), and batch predictions saved to CSV.

   

🔥 What this project does

This repo contains a single, interactive script that helps you build a robust multiple linear regression model from scratch:

✅ Loads a CSV dataset

🧼 Cleans data (drop irrelevant columns, handle missing values)

🧹 Removes outliers based on skewness-aware quantile trimming

📈 Checks linearity with quick scatterplots and applies log transforms (target first, then features as needed)

🧩 Handles multicollinearity using Variance Inflation Factor (VIF) and automatically drops high‑VIF features

🧪 Encodes categoricals with one‑hot/dummies

🎚️ Scales features with StandardScaler

🔄 Splits data into train/test and trains LinearRegression

📊 Evaluates on the test set with R²

📦 Generates predictions for a new CSV and saves results to file

📁 Repo structure
.
├── mlr_pipeline.py           # Main script (rename if you use another filename)
├── data/                     # (optional) place your CSVs here
└── README.md                 # You are here

🧰 Requirements

Python 3.9+

numpy

pandas

matplotlib

seaborn

scikit-learn

statsmodels

Install
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels

Note: The script uses interactive prompts and quick pop-up plots, so run it in an environment that can display matplotlib windows (e.g., VS Code, Jupyter, or a local terminal with GUI).

🚀 Usage

Place your training CSV somewhere accessible (e.g., data/train.csv).

Run the script:
python mlr_pipeline.py

Follow the prompts:

📁 Enter the sample data file name: → e.g., data/train.csv

🎯 Enter the target variable name: → e.g., SalePrice

🔢 Enter the feature vars (use "-" between names): → e.g., LotArea-OverallQual-OverallCond-YearBuilt

You’ll be shown scatter plots to visually check linearity and optionally apply log transforms (first to the target, then to non‑linear features).

The script will compute VIF and drop features with high multicollinearity (VIF > 10).

After training, it prints the R² score, the regression summary (intercept + coefficients), and prompts you for a new CSV to predict on.

Finally, it asks for a save name and writes predictions to your_name.csv.

🧪 Example
📁 Enter the sample data file name: data/house_prices.csv
🎯 Enter the target variable name: SalePrice
🔢 Enter the feature vars (use "-" between names): LotArea-OverallQual-YearBuilt-GarageCars

During linearity checks, you can choose:

1 = looks linear → keep

2 = not linear → script applies log to target first time, then to feature next time

The model trains, prints Test R², then asks for a file to score, e.g. data/house_prices_new.csv, and saves predictions to CSV.

📐 Key implementation details

Outlier policy (skew-aware):

-0.5 ≤ skew ≤ 0.5 → no trimming

0.5 < skew < 1 → trim at 99th percentile (upper tail)

skew ≥ 1 → trim at 98th percentile (upper tail)

-1 < skew < -0.5 → trim at 1st percentile (lower tail)

skew ≤ -1 → trim at 2nd percentile (lower tail)

Linearity handling:

Interactive scatterplots for each numeric feature vs. target

First detected non‑linearity triggers log(target); subsequent ones trigger log(feature)

Multicollinearity control:

Computes VIF over numeric features only

Iteratively drops the highest VIF feature until max VIF ≤ 10

Encoding & scaling:

pd.get_dummies(..., drop_first=False, dtype=int)

StandardScaler fitted on training features; reused for scoring new data

Evaluation:

Train/test split: 80/20, random_state=42

Metric: R² on test set

Predictions:

If target was logged, predictions are exponentiated back to original scale

Output CSV includes predicted_<target> column

📎 Tips

Ensure the new data (for predictions) has the same feature columns as the training set after all transformations. Mismatches will cause errors.

For strictly reproducible pipelines in production, consider refactoring this interactive flow into a non‑interactive class with a fit/transform/predict API and persistent artifacts (scaler, selected features, encoders).

🧭 Roadmap



🤝 Contributing

PRs are welcome! If you find an edge case (e.g., categorical targets, empty feature sets after VIF), open an issue with a reproducible snippet.

📜 License

MIT — feel free to use, modify, and share.

🙌 Acknowledgements

Built with ❤️ using pandas, scikit-learn, statsmodels, matplotlib, and seaborn.

[My CV].(https://github.com/yourusername/yourrepo/raw/main/CV.pdf)
