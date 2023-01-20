# machine-learning-project

# primary source for this README: jupyter-6-Supervised-Learning
Repository for running jupyter notebooks and keeping relevant files in one place

# notes made for previous plan to remove null values
check how to remove null values from dataframe

## notes
pandas 
.iloc() - locate by row, col indices
.loc() - locate by row index and col NAME

## Data Cleaning
### 2.13 Lecture
df.drop('Column name', axis=1)
    - where axies = 0 for rows, 1 for columns
    - drops referenced column from data frame
    - inplace=True argument to ensure column stays dropped.
df.drop(1,axis=0).reset_index()
    - new col with old indices
df.drop(1,axis=0).reset_index(drop=True,inplace=True)

df.copy

### 4.1 Lecture Data Sanity Checks - Part 1
df['columnname'].apply(type).value_counts()
    - this looks at and notes the values by type and then counts them

df['colname'] = df['colname'].replace('missing','inf'],np.nan)
    - replaces our specified strings 'missing' and 'inf' 
    -  with np.nan

df['colname'] = df['colname'].astype(float)
    - convert values to float

Review note: when we substitute np.nan in for strings the resulting data type is (if all the other entries are say float) float.

df.info()
    - rerunning this after data cleaning may result in cleaned columns type changing to, say, float.

Check length of each column 
Columns shorter than max col length means missing values as empty cells

#### Alternative approach - clean while loading:
##### using na_values to tell python which values it should consider as NaN
data_new = pd.read_csv('/content/drive/MyDrive/Python Course/Melbourne_Housing.csv',na_values=['missing','inf'])
- on load, above line automatically converts all missing and inf to nan so, running:
data_new['BuildingArea'].dtype
- gives 
dtype('float64')
as only float (and nan which seems to be treated as whatever the rest of the data types are)

#### Review note
data['BuildingArea'].unique()
- above line run before cleaning gives unique values in column as a numpy array
- so can inspect to find out which strings to remove.
# setup steps
python3 -m venv .venv
    - in bash
    - and on Windows
source .venv/bin/activate
    - in bash
source .venv/Scripts/activate
    - on Windows
/workspace/jupyter-6/.venv/bin/python -m pip install --upgrade pip
    - in GitPod
python3 -m pip install --upgrade pip
    - on Windows

pip install --upgrade pip
pip install jupyter notebook
pip install matplotlib
pip install pandas
pip install seaborn
pip install numpy
pip install scipy
pip install statsmodels
pip install -U scikit-learn
pip install ipykernel

Ctrl Shift P
Create New Jupyter Notebook
Save and name notebook
Paste in necessary code

Ctrl Shift P
Python: Select Interpreter
use Python version in ./.venv/bin/python

pip freeze > requirements.txt

pip install -r requirements.txt

## Add required files
auto-mpg.csv
## Extensions
Extension: Excel Viewer
    - for  viewing csv files in VSCode

## Debug
### prelim
per above
Python:Select Interpreter
3.10.9 (.venv)
### ipykernel bug
after running
pip install ipykernel
on running LinearRegression_HandsOn-1.ipynb
message appears saying:
it is necessary to install ipykernel
OK
installing ipykernel
Rerun
LinearRegression_HandsOn-1.ipynb

### pandas bug
after running
pip install pandas 
pandas not found

### Fix for previious 2 bugs
create new jupyter notebook using 
Ctrl Shift P
Create New Jupyter Notebook
# References
## previous repositories
jupyter-test
jupyter-repo-2
jupyter-3
- [Coder731/jupyter-5](https://github.com/Coder731/jupyter-5)
- [Coder731/jupyter-6-Supervised-Learning](https://github.com/Coder731/jupyter-6-Supervised-Learning)


# References Part2 / (MyGreatLearning, Colab, modules)
#### MyGreatLearning
##### pre scikit-learn
- [LMS - Hands_on_Notebook_Week3.ipynb](https://www.mygreatlearning.com/)
- [LMS - ENews_Express_Learner_Notebook%5BLow_Code_Version%5D.ipynb](https://www.mygreatlearning.com/)
- [LMS - abtest.csv](https://www.mygreatlearning.com/)
- [2.13 Pandas - Accessing and Modifying DataFrames (condition-based indexing)](https://www.mygreatlearning.com/)
#### scikit-learn
- [Supervised Learning - Foundations / Week 1 - Lecture Video Materials](https://www.mygreatlearning.com/)
    - [auto-mpg.csv used in 1.9 Linear Regression Hands-on](https://www.mygreatlearning.com/)

#### Colab
- Google Colab [mount drive](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=RWSJpsyKqHjH)

#### modules
##### matplotlib
###### matplotlib figure dimentions
- [Set plot dimensions matplotlib](https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib)

##### scipy
- [scipy - check version](https://blog.finxter.com/how-to-check-scipy-package-version-in-python/)


# References Part3 / (StackOverflow, Git, Tutorials and Repositories)
## StackOverflow
https://stackoverflow.com/questions/46419607/how-to-automatically-install-required-packages-from-a-python-script-as-necessary

## Git
### Gitpod
- [Gitpod docs prebuilds](https://www.gitpod.io/docs/configure/projects/prebuilds)
- [Gitpod docs workspaces](https://www.gitpod.io/docs/configure/workspaces/tasks)
- [Gitpod Prebuild](https://youtu.be/ZtlJ0PakUHQ?t=54)
### Git in VSCode
- [Git source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)

## Tutorials and Repositories

# References Part4 / (environments, Packages, and Statistics)
## environments
### local
- [Getting Full Directory Path in Python](https://www.youtube.com/watch?v=DQRSvg54bhM&ab_channel=Analyst%27sCorner)

Windows
Anaconda
conda create --name .cenv
y
conda activate .cenv

python3

not installed so Windows store opens
install Python 3.10

#### conda
##### virtual environment
- [conda.io](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)

#### python environment
`python3 -m venv .venv`
command was slow at first  but self-resolved
- search string: stuck on $ python3 -m venv .venv [setting up environment in virtaulenv using python3 stuck on ...](https://discuss.dizzycoding.com/setting-up-environment-in-virtaulenv-using-python3-stuck-on-setuptools-pip-wheel/)
- search string: installing collected packages stuck [why is the pip install process stuck on ''Installing collected packages" step?](https://stackoverflow.com/questions/54699197/why-is-the-pip-install-process-stuck-on-installing-collected-packages-step)

## Packages
### NumPy
### Pandas
### matplotlib
### scipy
### scipy.stats
### statsmodels
- [statsmodels.stats.proportion.proportions_ztest](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html)
### scikit-learn
#### Documentation
- [search string: sklearn](https://www.google.com/search?q=sklearn&oq=sklearn&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARAjGCcyBggCEAAYQzIGCAMQABhDMgYIBBAAGEMyBggFEAAYQzIGCAYQRRg8MgYIBxBFGDzSAQc3MzVqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [scikit-learn | Machine Learning in Python](https://scikit-learn.org/stable/)
- [Getting Started -- skikit-learn](https://scikit-learn.org/stable/getting_started.html)
- [Citing scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn)
- [User Guide](https://scikit-learn.org/stable/user_guide.html#user-guide)
- [Installing scikit-learn](https://scikit-learn.org/stable/install.html)
- Scikit-learn: Machine Learning in Python [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- redirects to https://scikit-learn.org/stable/ (link 2 in this section, above) [Source code, binaries, and documentation](http://scikit-learn.sourceforge.net)
### ipykernel
- [search string: ipykernel](https://www.google.com/search?q=ipykernel&oq=ipykernel&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIMCAgQABgUGIcCGIAEMgcICRAAGIAE0gEHNDUzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- pip install ipykernel [ipykernel 6.19.2](https://pypi.org/project/ipykernel/)
## Statistics

## pandas print statement
- [turn off automatic pandas data type output on print statment](https://stackoverflow.com/questions/29645153/remove-name-dtype-from-pandas-output-of-dataframe-or-series)

## naming arbitrary number of variables
- [used for first attempt at naming arbitrary number of variables](https://stackoverflow.com/questions/48372808/create-an-unknown-number-of-programmatically-defined-variables)
- [second attempt at naming arbitrary number of variables](https://pythonprinciples.com/ask/how-do-you-create-a-variable-number-of-variables/)