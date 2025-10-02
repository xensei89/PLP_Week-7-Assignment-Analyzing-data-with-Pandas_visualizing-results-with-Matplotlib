## README: Wine Quality Data Analysis Project 

This project analyzes the chemical properties of **red and white wine** to understand what factors influence wine quality. It combines two different datasets and uses common data science techniques to find and visualize key insights.

***

**Project Goal**

To load, clean, and analyze the red and white wine quality datasets using Python's **Pandas** library, and then visualize the results using **Matplotlib** and **Seaborn** to identify patterns in chemical properties that correlate with quality.


**Files in this Project**Source : UCI datasets : Wine Quality

| `winequality-red.csv` | Original raw data for red wines. |
| `winequality-white.csv` | Original raw data for white wines. |
| 'wine_analysis_final.py | The Python script
| `combined\_wine\_data.csv` | The final, cleaned dataset combining both red and white wines. |


**Key Findings 

The analysis focused on comparing the two types of wine and identifying chemical traits that predict a better quality score (from 3 to 9).

|| **Average Quality** | Slightly lower (approx. 5.64) | Slightly higher (approx. 5.88) | White wines, on average, received a better quality rating in the dataset. |
| **Volatile Acidity** | Much Higher | Much Lower | **Volatile Acidity** (a measure of acetic acid, or vinegar) is a key difference. High levels are less desirable, and red wines tend to have more. |
| **Residual Sugar** | Much Lower | Much Higher | White wines have significantly more **residual sugar**, aligning with them generally being sweeter. |
| **Alcohol Trend** | **Strong Positive Correlation** | **Strong Positive Correlation** | For *both* types, wines with **higher alcohol content** consistently received **higher average quality scores**. |


Tools Used

* **Python
* **Pandas:** Used for data loading, cleaning, and powerful numerical analysis (calculating means, standard deviations, etc.).
* **Matplotlib/Seaborn:** Used for creating professional and easy-to-read data visualizations (charts and graphs).
