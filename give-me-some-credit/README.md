# Kaggle Compeitition : Give Me Some Credit
- **Objective** : Predict probability of borrower might experience financial distress in the next two years.
- **Why?** : The [original competition](https://www.kaggle.com/c/GiveMeSomeCredit) ran several years ago. I want to try a few alternative approaches - which hopefully will get me into the top 100 of the private leaderboard.

#### What's been done so far?
- EDA (Counts, Outlier detection *and replacement*, etc.)
- Dealing with missing values
  - `MonthlyIncome` : built simple models to predict the missing values
  - `NumDependents` : used `Scikit-Learn`'s `Imputer`
- Dealing with class imbalance
- Some feature engineering

What's **not been done** so far :
*serious* feature engineering,
exotic ensembles

#### Project Structure
- `GiveMeSomeCredit.ipynb` - preliminary exploratory work, followed by few other notebooks. I have [shared a folder containing the notebooks rendered as HTML](https://app.box.com/s/4v8sg5vxdys43er4eo5ntue4gv9r9wqk) - if you encounter difficulties viewing the rendered version via GitHub.
- `feats.py`, `utils.py` & `ploty.py` : utilities used by the `.ipynb` notebook as well as other code
- `Vagrantfile`, `provision.sh`, `requirements.txt` : this project runs in a self-contained VirtualBox VM with the essential Python packages installed. Essential commands
  - To start it, `vagrant up`
  - To suspend VM, `vagrant suspend`
  - To resume work, `vagrant resume`
  - To destroy VM, `vagrant destroy`
