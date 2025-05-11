import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn as sk
from scipy import stats
import scikit_posthocs as sp

# Read in de-identified data as a pandas DataFrame
df = pd.read_csv("./data/cleaned_data.csv", index_col = 0)

# Define one day in terms of minutes. Limit scope to only include requests submitted within seven days (before/after) the assignment deadline.
one_day = 60*24
df = df[(df["delta (min)"] > -7*one_day) & (df["delta (min)"] < 7*one_day)]
df["days"] = df["days"].str[0].fillna(1).astype(int)

# Visualize the distribution of request times across all course and assignment types
sns.histplot(data = df, x = "delta (min)", bins = np.arange(-7*one_day, 7*one_day, one_day / 24))
plt.xlim(-7 * one_day, 7 * one_day)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.2)

plt.title("Extension requests relative to assignment deadline")
plt.xlabel("Time relative to assignment deadline (in minutes)")
plt.ylabel("Number of requests")

# Test the data for normality
stats.probplot(df["delta (min)"], dist = "norm", plot=plt)
plt.title("Quantile-Quantile plot of extension requests")
plt.xlabel("Theoretical quantiles")
plt.ylabel("Unscaled quantiles of sample data")

# A 3x3 cross-classification of distribution times
axs = sns.FacetGrid(df, row = "Class Type", col="Assignment Category", margin_titles=True)
axs.map(sns.histplot, "delta (min)", stat = "density", kde = True, bins = np.arange(-7*one_day, 7*one_day, one_day / 8));
plt.xlim(-7*one_day, 7*one_day)

# Extracting request times for each category in preparation for hypothesis testing.
lower_div = df[df["Class Type"] == "Lower Division"]["delta (min)"]
upper_div_core = df[df["Class Type"] == "Upper Division Core"]["delta (min)"]
upper_div_elective = df[df["Class Type"] == "Upper Division Elective"]["delta (min)"]

labs = df[df["Assignment Category"] == "Lab"]["delta (min)"]
hw = df[df["Assignment Category"] == "Homework"]["delta (min)"]
projects = df[df["Assignment Category"] == "Project"]["delta (min)"]

# Kruskal-Wallis test comparing course types
stats.kruskal(lower_div, upper_div_core, upper_div_elective)

# Kruskal-Wallis test comparing assignment types
stats.kruskal(labs, hw, projects)

# Dunn-Bonferroni test for pairwise course types
sp.posthoc_dunn(df, val_col = "delta (min)", group_col = "Class Type", p_adjust="bonferroni")

# Dunn-Bonferroni test for pairwise assignment types
sp.posthoc_dunn(df, val_col = "delta (min)", group_col = "Assignment Category")

# Cumulative distribution of requests across assignment types
assignments = df["Assignment Category"].unique()
plt.figure().set_figwidth(10)
for assignment in assignments:
  df_course = df[df["Assignment Category"] == assignment]
  data = df_course["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = assignment)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time relative to deadline (in minutes)")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative distribution of assignment requests across assignment types")
plt.legend(fontsize = 14)
plt.show()

# Cumulative distribution of requests across course types
divs = df["Class Type"].unique()
plt.figure().set_figwidth(10)
for div in divs:
  df_course = df[df["Class Type"] == div]
  data = df_course["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = div)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time relative to deadline (in minutes)")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative distribution of assignment requests across class types")
plt.legend(fontsize = 14)
plt.show()

# Box plot of days requested across course types
sns.boxplot(data = df, x = "Class Type", y = "days", hue = "Class Type")
plt.xlabel("Course type")
plt.ylabel("Number of days requested for extension")
plt.title("Number of days requested across course types")
plt.show()

# Box plot of days requested across assignment types
sns.boxplot(data = df, x = "Assignment Category", y = "days", hue = "Assignment Category")
plt.xlabel("Assignment type")
plt.ylabel("Number of days requested for extension")
plt.title("Number of days requested across assignment types")
plt.show()

# Cumulative distribution for Lab assignments across courses
classes = df["Class Name"].unique()
plt.figure().set_figwidth(10)
for course in classes:
  df_course = df[df["Class Name"] == course]
  data = df_course[df_course["Assignment Category"] == "Lab"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = course)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Labs)")
plt.legend()
plt.show()

# Cumulative distribution for Homework assignments across courses
plt.figure().set_figwidth(10)
for course in classes:
  df_course = df[df["Class Name"] == course]
  data = df_course[df_course["Assignment Category"] == "Homework"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = course)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Homework)")
plt.legend()
plt.show()

# Cumulative distribution for Projects across courses
plt.figure().set_figwidth(10)
for course in classes:
  df_course = df[df["Class Name"] == course]
  data = df_course[df_course["Assignment Category"] == "Project"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = course)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Project)")
plt.legend()
plt.show()

# Cumulative distribution for Lab assignments across course types
plt.figure().set_figwidth(10)
for div in divs:
  df_course = df[df["Class Type"] == div]
  data = df_course[df_course["Assignment Category"] == "Lab"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = div)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Lab)")
plt.legend()
plt.show()

# Cumulative distribution for Homework assignments across course types
plt.figure().set_figwidth(10)
for div in divs:
  df_course = df[df["Class Type"] == div]
  data = df_course[df_course["Assignment Category"] == "Homework"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = div)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Homework)")
plt.legend()
plt.show()

# Cumulative distribution for Projects across course types
plt.figure().set_figwidth(10)
for div in divs:
  df_course = df[df["Class Type"] == div]
  data = df_course[df_course["Assignment Category"] == "Project"]["delta (min)"]
  hist, bins = np.histogram(data, bins=100)
  # Calculate the CDF
  cdf = np.cumsum(hist) / np.sum(hist)
  # Plot the CDF
  plt.plot(bins[:-1], cdf, label = div)
for i in np.arange(-7*one_day, 7*one_day, one_day):
  plt.axvline(x = i, color = "red", linestyle = "--", alpha = 0.1)

plt.xlabel("Time until deadline (in minutes)")
plt.ylabel("Cumulative probability")
plt.title("Cumulative distribution of extension requests (Project)")
plt.legend()
plt.show()
