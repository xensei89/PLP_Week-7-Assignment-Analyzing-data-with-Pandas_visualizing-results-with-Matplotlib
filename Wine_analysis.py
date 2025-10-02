
# Wine Quality Analysis 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import scipy as sp      


def load_wine_data():
    """Load wine datasets with comprehensive error handling"""
    print("🔍 Attempting to load wine datasets...")

    # Try multiple possible file locations
    possible_paths = [
        "",
        "data/",
        "wine_quality/",
        "../"
    ]

    for base_path in possible_paths:
        red_path = os.path.join(base_path, "winequality-red.csv")
        white_path = os.path.join(base_path, "winequality-white.csv")

        if os.path.exists(red_path) and os.path.exists(white_path):
            print(f"✅ Found files in: {base_path if base_path else 'current directory'}")
            break
    else:
        print("❌ Could not find wine dataset files in common locations")
        print("Please ensure 'winequality-red.csv' and 'winequality-white.csv' are available")
        return None, None

    # Try different delimiters
    delimiters = [';', ',', '\t']

    for delimiter in delimiters:
        try:
            print(f"🔧 Trying delimiter: '{delimiter}'")
            red_wine = pd.read_csv(red_path, delimiter=delimiter)
            white_wine = pd.read_csv(white_path, delimiter=delimiter)

            # Validate basic structure
            if len(red_wine.columns) > 1 and len(white_wine.columns) > 1:
                print(f"✅ Successfully loaded with '{delimiter}' delimiter!")
                return red_wine, white_wine

        except Exception as e:
            continue

    print("❌ Failed to load files with common delimiters")
    return None, None


def validate_and_clean_data(red_wine, white_wine):
    """Comprehensive data validation and cleaning"""
    print("\n" + "=" * 60)
    print("DATA VALIDATION & CLEANING REPORT")
    print("=" * 60)

    if red_wine is None or white_wine is None:
        print("❌ No data to validate")
        return None, None

    # Basic info
    print(f"📊 Red Wine: {red_wine.shape[0]} rows, {red_wine.shape[1]} columns")
    print(f"📊 White Wine: {white_wine.shape[0]} rows, {white_wine.shape[1]} columns")

    # Clean column names (handle whitespace and case issues)
    red_wine.columns = red_wine.columns.str.strip().str.lower().str.replace(' ', '_')
    white_wine.columns = white_wine.columns.str.strip().str.lower().str.replace(' ', '_')

    print(f"\n🔧 Cleaned Red Wine columns: {red_wine.columns.tolist()}")
    print(f"🔧 Cleaned White Wine columns: {white_wine.columns.tolist()}")

    # Check for required columns
    required_columns = ['quality', 'alcohol', 'citric_acid', 'density']

    red_missing = [col for col in required_columns if col not in red_wine.columns]
    white_missing = [col for col in required_columns if col not in white_wine.columns]

    if red_missing:
        print(f"❌ Red Wine missing: {red_missing}")
        # Try to find similar columns
        for missing in red_missing:
            similar = [col for col in red_wine.columns if missing in col]
            if similar:
                print(f"   Similar columns found: {similar}")

    if white_missing:
        print(f"❌ White Wine missing: {white_missing}")
        for missing in white_missing:
            similar = [col for col in white_wine.columns if missing in col]
            if similar:
                print(f"   Similar columns found: {similar}")

    if red_missing or white_missing:
        return None, None

    # Handle missing values
    red_initial = len(red_wine)
    white_initial = len(white_wine)

    red_wine = red_wine.dropna()
    white_wine = white_wine.dropna()

    print(f"\n🧹 Data Cleaning Results:")
    print(f"   Red Wine: {red_initial} → {len(red_wine)} rows (removed {red_initial - len(red_wine)})")
    print(f"   White Wine: {white_initial} → {len(white_wine)} rows (removed {white_initial - len(white_wine)})")

    return red_wine, white_wine


def calculate_correlations(red_wine, white_wine):
    """Calculate proper correlations between quality and alcohol"""
    print("\n📈 Calculating correlations...")

    # Method 1: Pearson correlation between quality and alcohol
    red_corr = red_wine['quality'].corr(red_wine['alcohol'])
    white_corr = white_wine['quality'].corr(white_wine['alcohol'])

    # Method 2: Spearman correlation (rank-based, better for ordinal data)
    red_spearman = red_wine['quality'].corr(red_wine['alcohol'], method='spearman')
    white_spearman = white_wine['quality'].corr(white_wine['alcohol'], method='spearman')

    print(f"   Red Wine Pearson correlation: {red_corr:.3f}")
    print(f"   White Wine Pearson correlation: {white_corr:.3f}")
    print(f"   Red Wine Spearman correlation: {red_spearman:.3f}")
    print(f"   White Wine Spearman correlation: {white_spearman:.3f}")

    return {
        'red_pearson': red_corr,
        'white_pearson': white_corr,
        'red_spearman': red_spearman,
        'white_spearman': white_spearman
    }


def create_visualizations(red_wine, white_wine, red_alcohol, white_alcohol, correlations):
    """Create comprehensive visualizations"""
    print("\n🎨 Generating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # 1. Line Chart - Alcohol by Quality
    plt.figure(figsize=(14, 7))
    plt.plot(red_alcohol.index, red_alcohol.values, marker="o", linewidth=3,
             markersize=10, label="Red Wine", color='#8B0000', markeredgecolor='black')
    plt.plot(white_alcohol.index, white_alcohol.values, marker="s", linewidth=3,
             markersize=10, label="White Wine", color='#FFD700', markeredgecolor='black')

    plt.title("Average Alcohol Content by Wine Quality", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Wine Quality Score", fontsize=14, fontweight='bold')
    plt.ylabel("Average Alcohol (%)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(int(red_alcohol.index.min()), int(red_alcohol.index.max()) + 1))

    # Add correlation info to plot
    plt.text(0.02, 0.98,
             f"Red Wine Correlation: {correlations['red_pearson']:.3f}\nWhite Wine Correlation: {correlations['white_pearson']:.3f}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 2. Bar Chart - Citric Acid Comparison
    plt.figure(figsize=(10, 7))
    wine_types = ['Red Wine', 'White Wine']
    citric_means = [red_wine["citric_acid"].mean(), white_wine["citric_acid"].mean()]
    colors = ['#8B0000', '#FFD700']

    bars = plt.bar(wine_types, citric_means, color=colors, edgecolor='black',
                   alpha=0.8, linewidth=2)

    plt.title("Average Citric Acid in Red vs White Wines", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Wine Type", fontsize=14, fontweight='bold')
    plt.ylabel("Average Citric Acid (g/dm³)", fontsize=14, fontweight='bold')

    # Add value labels
    for bar, value in zip(bars, citric_means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.show()

    # 3. Histogram - Alcohol Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(red_wine["alcohol"], bins=20, color="#8B0000", alpha=0.7,
             edgecolor="black", density=True)
    ax1.set_title("Alcohol Distribution - Red Wine", fontweight='bold')
    ax1.set_xlabel("Alcohol (%)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)

    ax2.hist(white_wine["alcohol"], bins=20, color="#FFD700", alpha=0.7,
             edgecolor="black", density=True)
    ax2.set_title("Alcohol Distribution - White Wine", fontweight='bold')
    ax2.set_xlabel("Alcohol (%)")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. Scatter Plot - Alcohol vs Density
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(red_wine["alcohol"], red_wine["density"], alpha=0.6,
                color="#8B0000", edgecolors="k", s=50)
    ax1.set_title("Alcohol vs Density - Red Wine", fontweight='bold')
    ax1.set_xlabel("Alcohol (%)")
    ax1.set_ylabel("Density (g/cm³)")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(white_wine["alcohol"], white_wine["density"], alpha=0.6,
                color="#FFD700", edgecolors="k", s=50)
    ax2.set_title("Alcohol vs Density - White Wine", fontweight='bold')
    ax2.set_xlabel("Alcohol (%)")
    ax2.set_ylabel("Density (g/cm³)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. Quality Distribution
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    red_wine['quality'].value_counts().sort_index().plot(kind='bar', color='#8B0000',
                                                         alpha=0.7, edgecolor='black')
    plt.title('Red Wine Quality Distribution', fontweight='bold')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    white_wine['quality'].value_counts().sort_index().plot(kind='bar', color='#FFD700',
                                                           alpha=0.7, edgecolor='black')
    plt.title('White Wine Quality Distribution', fontweight='bold')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_insights(red_wine, white_wine, red_alcohol, white_alcohol, correlations):
    """Generate comprehensive insights"""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE WINE QUALITY ANALYSIS INSIGHTS")
    print("=" * 70)

    # Calculate additional metrics
    red_quality_range = f"{red_wine['quality'].min()}-{red_wine['quality'].max()}"
    white_quality_range = f"{white_wine['quality'].min()}-{white_wine['quality'].max()}"

    red_alcohol_range = f"{red_wine['alcohol'].min():.1f}%-{red_wine['alcohol'].max():.1f}%"
    white_alcohol_range = f"{white_wine['alcohol'].min():.1f}%-{white_wine['alcohol'].max():.1f}%"

    # Determine correlation strength
    def get_correlation_strength(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very weak"

    red_strength = get_correlation_strength(correlations['red_pearson'])
    white_strength = get_correlation_strength(correlations['white_pearson'])

    insights = f"""
1. **ALCOHOL & QUALITY RELATIONSHIP:**
   • Red Wine: {red_strength} {'positive' if correlations['red_pearson'] > 0 else 'negative'} correlation (r = {correlations['red_pearson']:.3f})
   • White Wine: {white_strength} {'positive' if correlations['white_pearson'] > 0 else 'negative'} correlation (r = {correlations['white_pearson']:.3f})
   • Highest alcohol content:
     - Red: Quality {red_alcohol.idxmax()} ({red_alcohol.max():.2f}%)
     - White: Quality {white_alcohol.idxmax()} ({white_alcohol.max():.2f}%)

2. **CITRIC ACID COMPARISON:**
   • Red Wine: {red_wine['citric_acid'].mean():.3f} g/dm³
   • White Wine: {white_wine['citric_acid'].mean():.3f} g/dm³
   • Difference: {abs(red_wine['citric_acid'].mean() - white_wine['citric_acid'].mean()):.3f} g/dm³
   • White wines contain {'more' if white_wine['citric_acid'].mean() > red_wine['citric_acid'].mean() else 'less'} citric acid

3. **DATASET CHARACTERISTICS:**
   • Red Wine: {len(red_wine):,} samples, Quality range: {red_quality_range}
   • White Wine: {len(white_wine):,} samples, Quality range: {white_quality_range}
   • Alcohol ranges:
     - Red: {red_alcohol_range}
     - White: {white_alcohol_range}

4. **QUALITY DISTRIBUTION:**
   • Most common Red Wine quality: {red_wine['quality'].mode().iloc[0]}
   • Most common White Wine quality: {white_wine['quality'].mode().iloc[0]}
   • Quality variability:
     - Red Wine std: {red_wine['quality'].std():.2f}
     - White Wine std: {white_wine['quality'].std():.2f}

5. **PRACTICAL IMPLICATIONS:**
   • {'✅' if correlations['red_pearson'] > 0.3 else '⚠️'} Alcohol content is {'a good' if correlations['red_pearson'] > 0.3 else 'not a strong'} predictor of red wine quality
   • {'✅' if correlations['white_pearson'] > 0.3 else '⚠️'} Alcohol content is {'a good' if correlations['white_pearson'] > 0.3 else 'not a strong'} predictor of white wine quality
   • The data suggests focusing on {'alcohol content' if max(correlations['red_pearson'], correlations['white_pearson']) > 0.4 else 'other factors'} for quality prediction
    """

    print(insights)

    # Additional statistical insights
    print("\n" + "=" * 50)
    print("📈 ADDITIONAL STATISTICAL INSIGHTS")
    print("=" * 50)

    print(f"Statistical Significance (t-test for alcohol means):")
    t_stat, p_value = stats.ttest_ind(red_wine['alcohol'], white_wine['alcohol'])
    print(f"• t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
    print(
        f"• {'Significant' if p_value < 0.05 else 'No significant'} difference in alcohol content between red and white wines")

    print(f"\nQuality Score Frequencies:")
    print(f"• Red Wine: {dict(red_wine['quality'].value_counts().sort_index())}")
    print(f"• White Wine: {dict(white_wine['quality'].value_counts().sort_index())}")


def main():
    """Main execution function"""
    print("🍷 WINE QUALITY ANALYSIS TOOL")
    print("=" * 50)

    try:
        # Load data
        red_wine, white_wine = load_wine_data()
        if red_wine is None:
            return

        # Validate and clean data
        red_wine, white_wine = validate_and_clean_data(red_wine, white_wine)
        if red_wine is None:
            return

        # Initial data exploration
        print("\n" + "=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)

        print("\n Red Wine Sample:")
        print(red_wine.head(3))
        print("\n White Wine Sample:")
        print(white_wine.head(3))

        print("\n Red Wine Statistics:")
        print(red_wine[['quality', 'alcohol', 'citric_acid', 'density']].describe())
        print("\n White Wine Statistics:")
        print(white_wine[['quality', 'alcohol', 'citric_acid', 'density']].describe())

        # Calculate average alcohol by quality
        print("\n Average Alcohol by Quality:")
        red_alcohol = red_wine.groupby("quality")["alcohol"].mean()
        white_alcohol = white_wine.groupby("quality")["alcohol"].mean()

        print("Red Wine:")
        print(red_alcohol)
        print("\nWhite Wine:")
        print(white_alcohol)

        # Calculate correlations
        correlations = calculate_correlations(red_wine, white_wine)

        # Create visualizations
        create_visualizations(red_wine, white_wine, red_alcohol, white_alcohol, correlations)

        # Generate insights
        generate_insights(red_wine, white_wine, red_alcohol, white_alcohol, correlations)

        print("\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure winequality-red.csv and winequality-white.csv are in the correct directory")
        print("2. Check that files are from the UCI Wine Quality dataset")
        print("3. Verify file permissions and encoding")
        print("4. Try placing files in the same directory as this script")


if __name__ == "__main__":
    main()
