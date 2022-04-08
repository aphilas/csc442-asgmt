## Task (summary)

Explore the conversion of some of the numeric attributes to nominal using relevant information you obtain from
credible references for the chronic kidney disease dataset. After these conversion, do the various ML algorithms perform better or worse? { KNN, J48, MLP }

## Ubuntu installation
```bash
# https://fracpete.github.io/python-weka-wrapper3/install.html#ubuntu

sudo apt install -y build-essential python3-dev graphviz graphviz-dev default-jdk
pip install -r requirements.txt

# to install 'in order' instead - python-javabridge fails if numpy is not installed
# cat requirements.txt | xargs pip install
```
## Usage
```bash
python main.py | tee output.txt
```
## Docs

[python-weka-wrapper Docs](https://fracpete.github.io/python-weka-wrapper3/index.html)