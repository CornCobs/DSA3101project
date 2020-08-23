# Web App with Streamlit

The easiest way is to use `Anaconda` package manager.

1. Start up `Anaconda Navigator`
2. Go to `Environments` Tab and create a new virtual environment
3. Activate the virtual environment
4. Open `VS Code` in `Home` Tab
5. Open the project directory. The project structure is as follows:
- `src`
  - `app.py`
- `data`
  - `DSA3101_Hackathon_Categories_Information.csv`
  - `DSA3101_Hackathon_Data.csv`
  - `DSA3101_Hackathon_Panelists_Demographics.xlsx`
6. From the terminal (Ctrl+Shift+\`) in VS Code, run `streamlit run src/app.py`.
You may have to download requirements with the command `pip install -r requirements.txt`